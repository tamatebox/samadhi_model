from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from model.samadhi import SamadhiCore


class SamadhiTrainer:
    """
    Samadhi Modelのための汎用トレーナー。
    画像(MNIST)、時系列(Wave)、言語(Embedding)など、あらゆるベクトルデータに対応。
    """

    def __init__(self, model: SamadhiCore, optimizer: optim.Optimizer, device: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer

        # デバイス自動判定 (Mac: mps, NVIDIA: cuda, Others: cpu)
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

    def train_step(self, clean_batch: torch.Tensor, noise_fn: Callable) -> float:
        """
        1バッチ分の学習ステップを実行
        Args:
            clean_batch: ノイズのない正解データ (Batch, Dim)
            noise_fn: 関数 f(tensor) -> noisy_tensor
        """
        clean_batch = clean_batch.to(self.device)

        # 1. 動的にノイズを注入 (Chaos Generation)
        noisy_batch = noise_fn(clean_batch).to(self.device)

        self.optimizer.zero_grad()

        # 2. Forward Pass
        # A. Search (Vitakka)
        s0, _ = self.model.vitakka_search(noisy_batch)

        # B. Refine (Vicara) - 学習用に勾配を維持しながら実行
        s_t = s0
        batch_stability_loss = torch.tensor(0.0, device=self.device)

        num_steps = self.model.config["refine_steps"]
        if num_steps > 0:
            for _ in range(num_steps):
                s_prev = s_t
                residual = self.model.refiner(s_t)
                s_t = 0.7 * s_t + 0.3 * residual
                # バッチ内の各サンプルの変化量(L2ノルム)を合計
                batch_stability_loss += torch.sum(torch.norm(s_t - s_prev, p=2, dim=1))

        s_final = s_t

        # C. Loss計算
        # 復元Loss: バッチ全体でMSEを計算
        batch_recon_loss = nn.MSELoss()(s_final, clean_batch)

        # 安定性Loss: バッチとステップで平均化
        if num_steps > 0:
            batch_stability_loss /= len(clean_batch) * num_steps

        # 3. Backward
        # 復元誤差 + 安定性ペナルティ
        total_loss = batch_recon_loss + 0.1 * batch_stability_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def fit(self, dataloader, noise_fn: Callable, epochs: int = 5):
        """
        エポックを回して学習を実行
        """
        self.model.train()
        loss_history = []

        print(f"\n--- Start Training ({epochs} epochs) ---")

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for batch_idx, batch_data in enumerate(dataloader):
                # DataLoaderが (data, label) を返す場合と dataのみの場合に対応
                if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                    clean_data = batch_data[0]  # 画像データなど
                else:
                    clean_data = batch_data  # Tensorそのまま

                # Flattenが必要な場合はここで行うか、noise_fnで行う
                # ここでは汎用性のため、入力次元に合わせてViewが必要なら呼び出し元で整形済みであることを期待する
                # (画像の場合は [B, 1, 28, 28] -> [B, 784] になっているべき)
                if clean_data.dim() > 2:
                    clean_data = clean_data.view(clean_data.size(0), -1)

                loss = self.train_step(clean_data, noise_fn)
                total_loss += loss
                count += 1

                if batch_idx % 50 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss:.4f}", end="")

            avg_loss = total_loss / count
            loss_history.append(avg_loss)
            print(f"\nEpoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        return loss_history
