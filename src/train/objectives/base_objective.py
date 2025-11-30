from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class BaseObjective(ABC):
    """
    Samadhi トレーニング目的の抽象基底クラスです。
    合計損失と個々の損失成分を計算するためのインターフェースを定義します。

    プロパティ:
        needs_vitakka (bool): Vitakka（探索プロセス）を必要とするかどうか。Falseの場合、Adapterの出力が直接潜在状態として扱われる。
        needs_vicara (bool): Vicara（浄化プロセス）を必要とするかどうか。Falseの場合、Vicaraはスキップされる。
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
        確率分布の正規化されたエントロピーを計算するためのヘルパー関数です。
        [0, 1] の値を返します。
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
        プローブ崩壊を防ぐために正規化された負荷分散損失を計算します。
        バッチ全体の平均プローブ使用量の分散にペナルティを与えます。
        [0, 1] の値を返します。
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
        合計損失を計算し、個々の損失成分の辞書を返します。

        引数:
            x (torch.Tensor): 元の入力データ。
            y (Optional[torch.Tensor]): ターゲットデータ（教師あり学習用）。
            s0 (torch.Tensor): Vitakka からの初期潜在状態。
            s_final (torch.Tensor): Vicara からの最終的に浄化された潜在状態。
            decoded_s_final (torch.Tensor): s_final にデコーダーを適用した出力。
            metadata (Dict[str, Any]): Vitakka からのメタデータ（例: プローブ確率）。
            num_refine_steps (int): Vicara 洗練ステップの数。

        戻り値:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - total_loss (torch.Tensor): 結合された損失。
                - loss_components (Dict[str, Any]): 個々の損失値の辞書。
        """
        pass
