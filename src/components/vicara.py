from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VicaraBase(nn.Module, ABC):
    """
    Vicāra (Sustained Application/Refinement) Component Base Class.

    役割: 初期状態 S0 を受け取り、再帰的プロセスによってノイズを除去し、状態を純化（Refine）する。
    思考の微細な段階（Fine-grained thinking）を担当。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]
        self.steps = config["refine_steps"]

        # Refiner Network (The "Phi" loop)
        # S_t -> S_{t+1} (residual)
        self.refiner = self._build_refiner()

    def _build_refiner(self) -> nn.Module:
        """
        Build the refiner network.
        Can be overridden by subclasses (e.g. for Latent Refiner with specific architecture).
        """
        return nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
            nn.Tanh(),  # State is bound to -1~1
        )

    def forward(
        self, s0: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
        """
        Vicara Process: Recursive Refinement.

        Args:
            s0: Initial State (Batch, Dim)
            context: Optional context from Vitakka (e.g., probs for weighted refinement)

        Returns:
            s_final: Converged State (Batch, Dim)
            trajectory: List of state vectors (for visualization)
            energies: List of energy values (stability loss)
        """
        s_t = s0.clone()

        # Logging containers
        # Note: Trajectory logging might be heavy for large batches, usually used for inference/debug
        trajectory = []
        energies = []

        # Initial state log (detach for cpu/numpy)
        if not self.training:
            trajectory.append(s_t.detach().cpu().numpy())

        for _ in range(self.steps):
            s_prev = s_t.clone()

            # Refinement Step (Abstracted for Standard vs Weighted)
            residual = self._refine_step(s_t, context)

            # Inertial Update (EMA)
            # s_new = 0.7 * s_old + 0.3 * residual
            s_t = 0.7 * s_t + 0.3 * residual

            # Compute Energy (Stability)
            # Batch-wise mean energy for simple logging, or keep individual
            # Here we compute mean for simple list logging
            dist = torch.norm(s_t - s_prev, dim=1)
            energy = dist.mean().item()
            energies.append(energy)

            if not self.training:
                trajectory.append(s_t.detach().cpu().numpy())

            # Early Stopping (Appanā) - Inference only
            if not self.training and energy < 1e-4:
                break

        return s_t, trajectory, energies

    @abstractmethod
    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """
        1ステップの純化計算。
        Standardでは単一のRefinerを通すが、Weightedでは複数のRefinerを混ぜるなどの拡張が可能。
        """
        pass


class StandardVicara(VicaraBase):
    """
    Standard Vicāra.
    単一の Refiner ネットワークを使用して純化を行う。
    Hard Vitakka と組み合わせるのが基本だが、Soft Vitakka と組み合わせても
    「入り口は曖昧だが、純化のルールは一つ」という構成で動作する。
    """

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        return self.refiner(s_t)


class WeightedVicara(VicaraBase):
    """
    Weighted Vicāra (Optional / Advanced).

    もし 'Phi' (Refiner) 自体を Probe ごとに持ちたい場合や、
    Probe の確信度に応じて Refiner の挙動を変えたい場合に使用する。

    今回は、設計計画の "Standard" に相当する、共有 Phi を想定しているため、
    StandardVicara と同じ実装で十分だが、将来的な拡張（Mixture of Experts Refinerなど）
    のためにクラスを分けておく。

    Current Implementation: Just calls shared refiner (Same as Standard).
    """

    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        # Future extension:
        # if self.multiple_refiners:
        #    return weighted_sum(refiners(s_t), context['probs'])
        return self.refiner(s_t)
