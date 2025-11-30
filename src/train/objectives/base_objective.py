from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class BaseObjective(ABC):
    """
    Abstract base class for Samadhi training objectives.
    Defines the interface for computing total loss and individual loss components.

    Properties:
        needs_vitakka (bool): Whether Vitakka (Search process) is required. If False, Adapter output is used directly as the initial latent state.
        needs_vicara (bool): Whether Vicara (Refinement process) is required. If False, Vicara is skipped.
    """

    needs_vitakka: bool = True
    needs_vicara: bool = True

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device) if device else self._get_default_device()

    def _get_default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Helper function to compute the NORMALIZED entropy of a probability distribution.
        Returns a value in [0, 1].
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
        n_probes = self.config["n_probes"]
        if n_probes > 1:
            max_entropy = torch.log(torch.tensor(n_probes, dtype=torch.float, device=self.device))
            return entropy / max_entropy
        else:
            return entropy

    def _compute_load_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the NORMALIZED Load Balancing Loss to prevent Probe Collapse.
        Penalizes the variance of the average probe usage across the batch.
        Returns a value in [0, 1].
        """
        mean_usage = probs.mean(dim=0)
        balance_loss = mean_usage.var()
        n_probes = self.config["n_probes"]
        if n_probes > 1:
            max_variance = (n_probes - 1) / (n_probes**2)
            return balance_loss / max_variance
        else:
            return balance_loss

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        s0: torch.Tensor,
        s_final: torch.Tensor,
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the total loss and returns a dictionary of individual loss components.

        Args:
            x (torch.Tensor): Original input data.
            y (Optional[torch.Tensor]): Target data (for supervised learning).
            s0 (torch.Tensor): Initial latent state from Vitakka.
            s_final (torch.Tensor): Final purified latent state from Vicara.
            decoded_s_final (torch.Tensor): Output from the decoder applied to s_final.
            metadata (Dict[str, Any]): Metadata from Vitakka (e.g., probe probabilities).
            num_refine_steps (int): Number of Vicara refinement steps.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - total_loss (torch.Tensor): The combined loss.
                - loss_components (Dict[str, Any]): Dictionary of individual loss values.
        """
        pass
