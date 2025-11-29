from typing import Optional, List, Dict, Tuple
import torch
import torch.optim as optim
from src.model.samadhi import SamadhiModel


class BaseSamadhiTrainer:
    """
    Base trainer class for the Samadhi Model.
    Provides common initialization, inference logic, and utilities.
    """

    def __init__(self, model: SamadhiModel, optimizer: optim.Optimizer, device: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer

        # Automatic device detection
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"Trainer initialized on device: {self.device}")

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Helper function to compute the NORMALIZED entropy of a probability distribution.
        Returns a value in [0, 1].
        """
        # Negative sum of p * log(p). Added 1e-9 for numerical stability.
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()

        # Normalize by log(n_probes) to range [0, 1]
        n_probes = self.model.config["n_probes"]
        # Use a small epsilon to avoid division by zero if n_probes=1
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
        # probs: (Batch, NumProbes)
        # 1. Calculate average usage of each probe in the batch
        mean_usage = probs.mean(dim=0)  # (NumProbes,)

        # 2. Calculate variance of usage
        balance_loss = mean_usage.var()

        # 3. Normalize by max possible variance
        # Max variance occurs when one probe has prob 1.0 and others 0.0
        # Var_max = (K-1)/K^2
        n_probes = self.model.config["n_probes"]
        if n_probes > 1:
            max_variance = (n_probes - 1) / (n_probes**2)
            return balance_loss / max_variance
        else:
            return balance_loss

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        Executes a single training step for one batch.
        To be implemented by subclasses.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor, Optional): Target data (for supervised learning only).
        """
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """
        Executes the training loop.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(self, dataloader) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Executes inference using the trained model.
        Implemented here as common logic.

        Returns:
            Tuple[List[torch.Tensor], List[Dict]]:
                - List of purified data (CPU Tensor)
                - List of inference logs (Dict or None)
        """
        self.model.eval()
        self.model.to(self.device)

        all_results = []  # Purified image data
        all_logs = []  # Log data

        print("Running inference...")
        with torch.no_grad():
            for batch_data in dataloader:
                # 1. Extract data (Tuple or Tensor)
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    data = batch_data[0]
                else:
                    data = batch_data

                data = data.to(self.device)

                # 3. Infer each sample in the batch
                # (Process one by one and list, rather than forward_sequence)
                for i in range(len(data)):
                    x_in = data[i : i + 1]  # (1, Dim) or (1, C, H, W)

                    # step_idx=0 (dummy)
                    out = self.model.forward_step(x_in, step_idx=0)

                    if out:
                        s_final, log = out
                        # Apply decoder to get the final output (image or vector)
                        final_output = self.model.decoder(s_final)
                        all_results.append(final_output.cpu())  # Save to CPU
                        all_logs.append(log)
                    else:
                        # Gate Closed (Rejected)
                        # Return zeros of the same size as input (or input itself) for failed noise removal.
                        all_results.append(torch.zeros_like(x_in).cpu())
                        all_logs.append(None)

        return all_results, all_logs
