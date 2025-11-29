from typing import Optional, List, Dict, Tuple
import torch
import torch.optim as optim
from src.model.samadhi import SamadhiModel


class BaseSamadhiTrainer:
    """
    Samadhi Modelのためのトレーナー基底クラス。
    共通の初期化処理、推論ロジック、ユーティリティを提供する。
    """

    def __init__(self, model: SamadhiModel, optimizer: optim.Optimizer, device: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer

        # デバイス自動判定
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
        """確率分布のエントロピーを計算するヘルパー関数"""
        # p * log(p) の和をマイナスしたもの。0log0対策で +1e-9
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        1バッチ分の学習ステップを実行。
        サブクラスで実装すること。
        Args:
            x (torch.Tensor): 入力データ。
            y (torch.Tensor, Optional): ターゲットデータ（教師あり学習の場合のみ）。
        """
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """
        学習ループを実行。
        サブクラスで実装すること。
        """
        raise NotImplementedError

    def predict(self, dataloader) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        学習済みモデルを使って推論を実行します。
        共通ロジックとしてここで実装。

        Returns:
            Tuple[List[torch.Tensor], List[Dict]]:
                - 純化されたデータのリスト (CPU Tensor)
                - 推論ログのリスト (Dict or None)
        """
        self.model.eval()
        # Explicitly set hard attention for inference
        # Vitakka now handles mode switching internally, so no need to rebuild the instance.
        self.model.config["attention_mode"] = "hard"

        self.model.to(self.device)

        all_results = []  # 純化された画像データ
        all_logs = []  # ログデータ

        print("Running inference...")
        with torch.no_grad():
            for batch_data in dataloader:
                # 1. データの取り出し (Tuple or Tensor)
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    data = batch_data[0]
                else:
                    data = batch_data

                data = data.to(self.device)

                # 2. Flatten (画像の場合)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)

                # 3. バッチ内の各サンプルを推論
                # (forward_sequenceではなく、1つずつ処理してリスト化する)
                for i in range(len(data)):
                    x_in = data[i : i + 1]  # (1, Dim)

                    # step_idx=0 (ダミー)
                    out = self.model.forward_step(x_in, step_idx=0)

                    if out:
                        s_final, log = out
                        all_results.append(s_final.cpu())  # CPUに戻して保存
                        all_logs.append(log)
                    else:
                        # Gate Closed (棄却)
                        # ノイズ除去失敗として、入力と同じサイズのゼロ(または入力そのもの)を返す
                        all_results.append(torch.zeros_like(x_in).cpu())
                        all_logs.append(None)

        return all_results, all_logs
