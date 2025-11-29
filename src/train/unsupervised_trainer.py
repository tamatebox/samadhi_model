from typing import Optional
import torch
import torch.nn.functional as F
from src.train.base_trainer import BaseSamadhiTrainer


class UnsupervisedSamadhiTrainer(BaseSamadhiTrainer):
    """
    Samadhi Modelのための教師なし学習トレーナー。
    入力データ(x)のみを用いて学習を行う。
    ターゲットデータ(y)は使用しない。
    Stability Loss と Entropy Loss のみを最小化する。
    これにより、モデルは「自発的に安定するアトラクタ（概念）」を自己組織化する。
    """

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        1バッチ分の学習ステップを実行
        Args:
            x (torch.Tensor): 入力データ (Batch, Dim)
            y (torch.Tensor, Optional): 無視される
        """
        x = x.to(self.device)
        # y is ignored

        self.optimizer.zero_grad()

        # ====================================================
        # 2. Forward Pass
        # ====================================================

        # --- A. Search (Vitakka) ---
        # s0 を取得
        s0, metadata = self.model.vitakka_search(x)

        # Entropy Loss用の確率分布計算
        # Metadataから取得する
        probs = metadata["probs"]

        # --- B. Refine (Vicara) ---
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)
        num_steps = self.model.config["refine_steps"]

        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.vicara.refine_step(s_t, metadata)
                # 慣性更新
                s_t = 0.7 * s_t + 0.3 * residual

                # バッチ内の各サンプルの変化量(L2ノルム)を合計
                batch_stability_loss += torch.norm(s_t - s_prev, p=2, dim=1).sum()

        # ====================================================
        # 3. Loss Calculation
        # ====================================================

        # (1) Reconstruction Loss は使用しない (正解がないため)

        # (2) 安定性誤差 (Stability Loss): 心が不動になったか
        if num_steps > 0:
            batch_stability_loss = batch_stability_loss / (len(x) * num_steps)

        # (3) エントロピー誤差 (Entropy Loss): 迷わず選んだか
        entropy_loss = self._compute_entropy(probs)

        # (Optional) Sparsity Loss or Latent Regularization could be added here
        # to prevent mode collapse (e.g., all inputs mapping to same probe).
        # For now, we rely on Entropy Loss and diverse input data.

        # --- Total Loss ---
        stability_coeff = self.model.config.get("stability_coeff", 0.01)
        entropy_coeff = self.model.config.get("entropy_coeff", 0.1)

        # 教師なし学習では Stability と Entropy のバランスが全て
        total_loss = (stability_coeff * batch_stability_loss) + (entropy_coeff * entropy_loss)

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def fit(self, dataloader, epochs: int = 5, attention_mode: str = "soft"):
        """
        エポックを回して教師なし学習を実行
        dataloaderは (x) または (x, y) を返すが、xのみを使用する。
        """
        self.model.train()
        self.model.config["attention_mode"] = attention_mode

        loss_history = []

        print(f"\n--- Start Unsupervised Training ({epochs} epochs) ---")
        print(f"Device: {self.device}")
        print(
            f"Params: Stability={self.model.config.get('stability_coeff', 0.01)}, Entropy={self.model.config.get('entropy_coeff', 0.1)}"
        )

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # DataLoaderの形式対応
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x_batch = batch_data[0]
                    # y_batch (index 1) is ignored
                else:
                    x_batch = batch_data

                # Flatten対応
                if x_batch.dim() > 2:
                    x_batch = x_batch.view(x_batch.size(0), -1)

                loss = self.train_step(x_batch)  # y is optional
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
