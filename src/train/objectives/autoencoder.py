from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from src.train.objectives.base_objective import BaseObjective


class AutoencoderObjective(BaseObjective):
    """
    オートエンコーダの目的関数。再構成損失のみを計算します。
    Vitakka と Vicara のプロセスはスキップされます。
    """

    needs_vitakka: bool = False
    needs_vicara: bool = False

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        self.recon_loss_fn = nn.MSELoss()

    def compute_loss(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        s0: torch.Tensor,  # Vitakkaをスキップする場合、これはAdapterの出力となる
        s_final: torch.Tensor,  # Vicaraをスキップする場合、これはs0と同じ
        decoded_s_final: torch.Tensor,
        metadata: Dict[str, Any],
        num_refine_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # オートエンコーダの場合、s_finalはAdapterの出力z（すなわちs0）であり、
        # decoded_s_finalはそれをデコードした結果。ターゲットは元の入力x。
        recon_loss = self.recon_loss_fn(decoded_s_final, x)

        total_loss = recon_loss

        loss_components = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "stability_loss": 0.0,  # スキップされるため0
            "entropy_loss": 0.0,  # スキップされるため0
            "balance_loss": 0.0,  # スキップされるため0
        }

        return total_loss, loss_components
