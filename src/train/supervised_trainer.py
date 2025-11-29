from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.train.base_trainer import BaseSamadhiTrainer


class SupervisedSamadhiTrainer(BaseSamadhiTrainer):
    """
    Samadhi Modelのための教師あり学習トレーナー。
    入力データ(x)とターゲットデータ(y)のペアを用いて学習を行う。
    Reconstruction Loss を使用する点が特徴。
    """

    def train_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> float:
        """
        1バッチ分の学習ステップを実行
        Args:
            x (torch.Tensor): ノイズ付き入力データ (Batch, Dim)
            y (torch.Tensor): ノイズのない正解データ (Batch, Dim)
        """
        if y is None:
            raise ValueError("Target 'y' cannot be None for SupervisedSamadhiTrainer.")

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # ====================================================
        # 2. Forward Pass
        # ====================================================

        # --- A. Search (Vitakka) ---
        # s0 を取得 (バッチ対応済み前提)
        s0, metadata = self.model.vitakka_search(x)

        # Entropy Loss用の確率分布計算
        # Vitakka内部と同じ計算を行い、プローブ選択の「迷い」を数値化する
        probs = metadata["probs"]

        # --- B. Refine (Vicara) ---
        # 学習用に勾配を維持しながら実行
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)
        num_steps = self.model.config["refine_steps"]

        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.vicara.refine_step(s_t, metadata)
                # 慣性更新 (0.7 / 0.3 のバランス)
                s_t = 0.7 * s_t + 0.3 * residual

                # バッチ内の各サンプルの変化量(L2ノルム)を合計
                batch_stability_loss += torch.norm(s_t - s_prev, p=2, dim=1).sum()

        s_final = s_t
        decoded_s_final = self.model.decoder(s_final)

        # ====================================================
        # 3. Loss Calculation
        # ====================================================

        # (1) 復元誤差 (Reconstruction Loss): 正解に近づいたか
        # 教師あり学習の核心：純化結果がターゲットと一致することを目指す
        recon_loss = nn.MSELoss()(decoded_s_final, y)

        # (2) 安定性誤差 (Stability Loss): 心が不動になったか
        if num_steps > 0:
            # バッチサイズとステップ数で正規化
            batch_stability_loss = batch_stability_loss / (len(x) * num_steps)

        # (3) エントロピー誤差 (Entropy Loss): 迷わず選んだか
        entropy_loss = self._compute_entropy(probs)

        # --- Total Loss ---
        # 係数をConfigから取得 (なければデフォルト値)
        stability_coeff = self.model.config.get("stability_coeff", 0.01)
        entropy_coeff = self.model.config.get("entropy_coeff", 0.1)

        total_loss = recon_loss + (stability_coeff * batch_stability_loss) + (entropy_coeff * entropy_loss)

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def fit(self, dataloader, epochs: int = 5, attention_mode: str = "soft"):
        """
        エポックを回して教師あり学習を実行
        dataloaderは (x, y) のペアを返すことを想定。
        """
        self.model.train()
        # Explicitly set soft attention for training
        # Vitakka now handles mode switching internally, so no need to rebuild the instance.
        self.model.config["attention_mode"] = attention_mode

        loss_history = []

        print(f"\n--- Start Supervised Training ({epochs} epochs) ---")
        print(f"Device: {self.device}")
        print(
            f"Params: Stability={self.model.config.get('stability_coeff', 0.01)}, Entropy={self.model.config.get('entropy_coeff', 0.1)}"
        )

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # DataLoaderの形式対応: (x, y, ...)
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    x_batch = batch_data[0]
                    y_batch = batch_data[1]
                else:
                    # DataLoaderが単一のテンソルを返す場合はエラー（教師ありなので）
                    raise ValueError("DataLoader must return (input, target) pairs for SupervisedSamadhiTrainer.")

                loss = self.train_step(x_batch, y_batch)
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
