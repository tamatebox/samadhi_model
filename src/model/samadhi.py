from typing import Dict, Tuple, List, Optional, Any
import torch
import torch.nn as nn

# Import components
from src.components.vitakka import Vitakka
from src.components.vicara import VicaraBase, StandardVicara, WeightedVicara


class SamadhiModel(nn.Module):
    """
    Samadhi Model (Deep Convergence Architecture).

    Composed of:
    1. Vitakka (Search Component): Finds the intent/concept.
    2. Vicara (Refinement Component): Purifies the state.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]

        # 1. Initialize Vitakka (Search)
        self.vitakka = Vitakka(config)

        # 2. Initialize Vicara (Refinement)
        self.vicara = self._build_vicara(config)

        # 3. Output Decoder (Expression)
        self.decoder = self._build_decoder()

        # History Log (Citta-santāna)
        self.history_log: List[Dict] = []

    def _build_vicara(self, config: Dict[str, Any]) -> VicaraBase:
        # Currently we use StandardVicara as default.
        # Future config can switch to WeightedVicara if needed.
        return StandardVicara(config)

    def _build_decoder(self) -> nn.Module:
        """
        Build the decoder network.
        Simplified to Identity as the decoder is not a learned component
        in this configuration, and final state is directly compared to target.
        """
        return nn.Identity()

    def load_probes(self, pretrained_probes: torch.Tensor):
        """Delegate probe loading to Vitakka."""
        self.vitakka.load_probes(pretrained_probes)

    @property
    def probes(self):
        """Access probes via Vitakka (for convenience/compatibility)."""
        return self.vitakka.probes

    @property
    def refiner(self):
        """Access refiner via Vicara (for compatibility)."""
        return self.vicara.refiner

    def forward_step(self, x_input: torch.Tensor, step_idx: int) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Single Time-Step Execution (Search -> Refine -> Log).

        Args:
            x_input: (1, Dim) - Single input for sequential processing
            step_idx: Current step index

        Returns:
            s_final: Converged state
            full_log: Logs
        """
        # 1. Vitakka (Search)
        s0, meta = self.vitakka(x_input)

        # Check Gate (using the first item in batch, assuming batch_size=1 for forward_step loop)
        is_gate_open = meta["gate_open"]  # This might be a tensor if batch > 1
        if isinstance(is_gate_open, torch.Tensor):
            is_gate_open = is_gate_open.item()

        if not is_gate_open:
            return None  # Gate Closed

        # 2. Vicara (Refinement)
        # Pass metadata as context (e.g. for weighted refinement in future)
        s_final, trajectory, energies = self.vicara(s0, context=meta)

        # 3. Dynamics (Meta-Cognition)
        # Extract single item from batch metadata for logging
        winner_id = meta["winner_id"].item() if isinstance(meta["winner_id"], torch.Tensor) else meta["winner_id"]

        # Get label if available
        labels = self.config.get("labels", [])
        winner_label = labels[winner_id] if isinstance(winner_id, int) and winner_id < len(labels) else str(winner_id)

        probe_log = {
            "winner_id": winner_id,
            "winner_label": winner_label,
            "confidence": (
                meta["confidence"].item() if isinstance(meta["confidence"], torch.Tensor) else meta["confidence"]
            ),
            "raw_score": (
                meta["raw_score"].item() if isinstance(meta["raw_score"], torch.Tensor) else meta["raw_score"]
            ),
            "gate_open": is_gate_open,
        }

        dynamics = self._compute_dynamics(probe_log)

        full_log = {
            "step": step_idx,
            "probe_log": probe_log,
            "dynamics": dynamics,
            "energies": energies,
            "s_norm": torch.norm(s_final).item(),
            # "trajectory": trajectory # Optional: can be large
        }

        self.history_log.append(full_log)

        return s_final, full_log

    def _compute_dynamics(self, current_log: Dict) -> Optional[Dict]:
        """Compute state transition dynamics."""
        if not self.history_log:
            return None

        prev_log = self.history_log[-1]["probe_log"]

        if current_log["winner_id"] == prev_log["winner_id"]:
            trans_type = "Sustain"
        else:
            trans_type = "Shift"

        # Label resolution if labels exist in config
        labels = self.config.get("labels", [])
        curr_label = (
            labels[current_log["winner_id"]]
            if current_log["winner_id"] < len(labels)
            else str(current_log["winner_id"])
        )
        prev_label = (
            labels[prev_log["winner_id"]] if prev_log["winner_id"] < len(labels) else str(prev_log["winner_id"])
        )

        return {
            "from": prev_label,
            "to": curr_label,
            "type": trans_type,
            "confidence_delta": current_log["confidence"] - prev_log["confidence"],
        }

    def vitakka_search(self, x: torch.Tensor):
        """Compatibility method for Trainer."""
        return self.vitakka(x)
