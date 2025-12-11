"""
StandardVipassana: Simple trajectory encoder with confidence monitoring.

Uses mean/variance aggregation across the trajectory to produce
a context vector and trust score.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from satipatthana.components.vipassana.base import BaseVipassana
from satipatthana.core.santana import SantanaLog
from satipatthana.configs.vipassana import StandardVipassanaConfig


class StandardVipassana(BaseVipassana):
    """
    Standard Vipassana using mean/variance trajectory aggregation.

    Extracts features from the convergence trajectory:
    - Position: Final converged state
    - Velocity: Movement from initial to final state
    - Smoothness: Inverse of energy variance (lower = smoother)

    The trust score reflects:
    - Process quality: Fewer steps, lower energy = smoother convergence
    - Semantic validity: Proximity to known concepts (Probes) = higher trust
    """

    def __init__(self, config: StandardVipassanaConfig = None):
        if config is None:
            config = StandardVipassanaConfig()
        super().__init__(config)

        self.hidden_dim = config.hidden_dim
        self.context_dim = config.context_dim

        # Will be initialized on first forward pass when dim is known
        self._encoder = None
        self._trust_head = None
        self._input_dim = None

    def _build_networks(self, state_dim: int):
        """Build encoder networks once state dimension is known."""
        # Feature vector for context: [s_star (dim), velocity_norm (1), avg_energy (1)]
        feature_dim = state_dim + 2
        self._input_dim = state_dim

        self._encoder = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.context_dim),
        )

        # Trust head uses trajectory quality + semantic features + grounding features
        # - velocity: convergence speed
        # - avg_energy: trajectory smoothness
        # - min_dist: distance from S* to nearest Probe (familiarity)
        # - entropy: ambiguity across Probes
        # - s0_min_dist: distance from s0 to nearest Probe (initial OOD degree)
        # - drift_magnitude: ||S* - s0|| (convergence drift)
        trust_feature_dim = 6  # velocity + avg_energy + min_dist + entropy + s0_min_dist + drift
        self._trust_head = nn.Sequential(
            nn.Linear(trust_feature_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _compute_pairwise_distances(self, points: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise L2 distances (MPS-compatible).

        Args:
            points: Points tensor (Batch, Dim)
            probes: Probe vectors (N_probes, Dim)

        Returns:
            dists: Pairwise distances (Batch, N_probes)
        """
        # ||s - p||^2 = ||s||^2 + ||p||^2 - 2 * s @ p^T
        points_sq = (points**2).sum(dim=1, keepdim=True)  # (B, 1)
        probes_sq = (probes**2).sum(dim=1, keepdim=True).T  # (1, K)
        cross_term = torch.mm(points, probes.T)  # (B, K)
        dists_sq = points_sq + probes_sq - 2 * cross_term  # (B, K)
        return torch.sqrt(dists_sq.clamp(min=1e-9))  # (B, K)

    def _compute_semantic_features(
        self, s_star: torch.Tensor, probes: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute semantic features from S* and Probes.

        Args:
            s_star: Converged state (Batch, Dim)
            probes: Probe vectors (N_probes, Dim) or None

        Returns:
            min_dist: Distance to nearest Probe (Batch, 1) - "Familiarity"
            entropy: Probe distribution entropy (Batch, 1) - "Ambiguity"
        """
        batch_size = s_star.size(0)
        device = s_star.device
        dtype = s_star.dtype

        if probes is None:
            # Fallback: return neutral values when probes not available
            return (
                torch.zeros(batch_size, 1, device=device, dtype=dtype),
                torch.zeros(batch_size, 1, device=device, dtype=dtype),
            )

        # Compute pairwise L2 distances (MPS-compatible)
        dists = self._compute_pairwise_distances(s_star, probes)

        # (a) Minimum distance to nearest Probe ("Familiarity")
        # Lower = more familiar with known concepts
        min_dist, _ = torch.min(dists, dim=1, keepdim=True)  # (Batch, 1)

        # (b) Entropy of Probe distribution ("Ambiguity")
        # Convert distances to probabilities via softmax(-dist)
        # Higher entropy = more uncertain about which concept it belongs to
        probs = F.softmax(-dists, dim=1)  # (Batch, N_probes)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1, keepdim=True)  # (Batch, 1)

        return min_dist, entropy

    def _compute_grounding_features(
        self, s_star: torch.Tensor, s0: torch.Tensor, probes: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute grounding features for OOD detection.

        These features capture information BEFORE Vicara convergence,
        which is essential for detecting OOD inputs that would otherwise
        be "pulled" to familiar attractor regions.

        Args:
            s_star: Converged state (Batch, Dim)
            s0: Initial state from Vitakka (Batch, Dim)
            probes: Probe vectors (N_probes, Dim) or None

        Returns:
            s0_min_dist: Distance from s0 to nearest Probe (Batch, 1) - "Initial OOD degree"
            drift_magnitude: ||S* - s0|| (Batch, 1) - "Convergence drift"
        """
        batch_size = s_star.size(0)
        device = s_star.device
        dtype = s_star.dtype

        # Drift magnitude: how far did Vicara move the state?
        drift_magnitude = torch.norm(s_star - s0, dim=1, keepdim=True)

        if probes is None:
            return (
                torch.zeros(batch_size, 1, device=device, dtype=dtype),
                drift_magnitude,
            )

        # s0's distance to nearest Probe (OOD degree BEFORE convergence)
        s0_dists = self._compute_pairwise_distances(s0, probes)
        s0_min_dist, _ = torch.min(s0_dists, dim=1, keepdim=True)

        return s0_min_dist, drift_magnitude

    def forward(
        self,
        s_star: torch.Tensor,
        santana: SantanaLog,
        probes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze the thinking process and produce context vector and trust score.

        Args:
            s_star: Converged state tensor (Batch, Dim)
            santana: SantanaLog containing the thinking trajectory
            probes: Probe vectors from Vitakka (N_probes, Dim), optional

        Returns:
            v_ctx: Context vector (Batch, context_dim) - embedding of "doubt"
            trust_score: Confidence tensor (Batch, 1) for external control
        """
        batch_size, state_dim = s_star.shape
        device = s_star.device
        dtype = s_star.dtype

        # Lazy initialization of networks
        if self._encoder is None or self._input_dim != state_dim:
            self._build_networks(state_dim)
            self._encoder = self._encoder.to(device)
            self._trust_head = self._trust_head.to(device)

        # Extract trajectory features
        num_steps = len(santana)
        initial_state = santana.get_initial_state()

        # Compute velocity (distance from initial to final state)
        if initial_state is not None:
            velocity = torch.norm(s_star - initial_state, dim=1, keepdim=True)
        else:
            velocity = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # Compute per-sample average energy from trajectory states
        # Energy = ||s_t - s_{t-1}||^2 summed over steps
        if num_steps >= 2:
            states_tensor = santana.to_tensor()  # (num_steps, batch_size, dim)
            # Compute state differences: s_t - s_{t-1}
            state_diffs = states_tensor[1:] - states_tensor[:-1]  # (num_steps-1, batch, dim)
            # Per-sample energy: sum of squared norms
            per_sample_energy = (state_diffs**2).sum(dim=2).sum(dim=0)  # (batch,)
            avg_energy_tensor = (per_sample_energy / (num_steps - 1)).unsqueeze(1)  # (batch, 1)
        else:
            avg_energy_tensor = torch.zeros(batch_size, 1, device=device, dtype=dtype)

        # Compute semantic features from S* and Probes
        min_dist, entropy = self._compute_semantic_features(s_star, probes)

        # Compute grounding features (s0-based OOD detection)
        s0 = initial_state if initial_state is not None else s_star
        s0_min_dist, drift_magnitude = self._compute_grounding_features(s_star, s0, probes)

        # Build feature vector for context: [s_star, velocity_norm, avg_energy]
        features = torch.cat([s_star, velocity, avg_energy_tensor], dim=1)

        # Encode to context vector
        v_ctx = self._encoder(features)

        # Compute trust score from:
        # - Process quality: velocity, avg_energy (how smoothly did it converge?)
        # - Semantic validity: min_dist, entropy (is the result in known territory?)
        # - Grounding: s0_min_dist, drift (was input OOD before convergence?)
        # Use log1p to normalize scale while preserving per-sample differences
        trust_features = torch.cat(
            [
                torch.log1p(velocity),
                torch.log1p(avg_energy_tensor),
                torch.log1p(min_dist),
                entropy,  # Already normalized (0 to log(K))
                torch.log1p(s0_min_dist),  # Key for OOD detection
                torch.log1p(drift_magnitude),  # Large drift = suspicious
            ],
            dim=1,
        )  # (Batch, 6)
        trust_score = self._trust_head(trust_features)  # (Batch, 1)

        return v_ctx, trust_score
