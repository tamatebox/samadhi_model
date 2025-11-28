from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamadhiCore(nn.Module):
    """
    Samadhi Model (Deep Convergence Architecture) Core Engine.

    生成（Divergence）ではなく、収束（Convergence）を目的とした再帰型アテンションモデル。
    入力ストリームから「意図」を検索(Vitakka)し、ノイズを遮断して状態を純化(Vicāra)します。

    Attributes:
        config (dict): モデルのハイパーパラメータおよび設定を含む辞書。
        dim (int): 内部状態ベクトルの次元数。
        probes (nn.Parameter): 概念プローブ（学習可能な基底ベクトル群）。
        refiner (nn.Sequential): 状態純化を行う非線形オートエンコーダ。
        history_log (list): 推論の履歴（Citta-santāna）を保持するリスト。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        モデルを初期化します。

        Args:
            config (dict): 以下のキーを含む設定辞書。
                - "dim" (int): モデルの次元数 (例: 64)
                - "n_probes" (int): プローブの数
                - "refine_steps" (int): 純化ループの最大回数
                - "softmax_temp" (float): 側方抑制の温度パラメータ
                - "gate_threshold" (float): ゲートを開くための類似度閾値 (0.0~1.0)
                - "labels" (List[str]): 各プローブに対応するラベル名のリスト
        """
        super().__init__()
        self.config = config
        self.dim = config["dim"]

        # --- A. Vitakka Module (Search & Probing) ---
        # 概念プローブの定義
        # 初期状態はランダムだが、正規化してコサイン類似度計算に適した形にする
        self.probes = nn.Parameter(torch.randn(config["n_probes"], self.dim))
        self._normalize_probes()

        # --- B. Vicāra Module (Recurrent Refinement) ---
        # 純化関数 (Refinement Function)
        # 情報を圧縮・展開することで、本質的特徴量のみを残す
        self.refiner = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, self.dim),
            nn.Tanh(),  # 状態を -1 ~ 1 の範囲に安定させる
        )

        # 履歴ログ (Citta-santāna / Stream of Consciousness)
        self.history_log: List[Dict] = []

    def _normalize_probes(self):
        """プローブベクトルをL2正規化します（内部利用）。"""
        with torch.no_grad():
            self.probes.div_(torch.norm(self.probes, dim=1, keepdim=True))

    def vitakka_search(self, x_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        [Search Phase] 入力に対してプローブを照射し、初期アトラクタを決定します。

        入力ベクトルと各プローブとのコサイン類似度を計算し、
        閾値(gate_threshold)を超えた場合のみ、その「意図」を採用して初期状態 S0 を生成します。

        Args:
            x_input (torch.Tensor): 入力ベクトル (Shape: [B, dim])

        Returns:
            s0 (torch.Tensor): 初期化された状態ベクトル (Shape: [B, dim])。
            metadata (dict): バッチ内の各アイテムに対応するメタデータ。
                             各キーはテンソルまたはリストを保持します。
                             例: {"winner_id": (B,) tensor, "gate_open": (B,) boolean tensor, ...}
        """
        # 1. 入力の正規化 & 共鳴スコアの計算
        x_norm = F.normalize(x_input, p=2, dim=1)
        raw_scores = torch.matmul(x_norm, self.probes.T)  # (B, n_probes)

        # 2. 絶対評価によるゲーティング
        max_raw_score, winner_idx = torch.max(raw_scores, dim=1)  # (B,), (B,)
        is_gate_open = max_raw_score > self.config["gate_threshold"]  # (B,)

        # 3. 相対評価による確信度算出 (Softmax with Temperature)
        w_hat = F.softmax(raw_scores / self.config["softmax_temp"], dim=1)  # (B, n_probes)
        confidence = w_hat.gather(1, winner_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # 4. 初期状態 S0 の生成
        winner_probes = self.probes[winner_idx]  # (B, Dim)
        s0_resonant = x_input * winner_probes
        s0 = torch.where(is_gate_open.unsqueeze(1), s0_resonant, torch.zeros_like(x_input))

        # ゲートが閉じたアイテムの確信度を0に設定
        confidence = torch.where(is_gate_open, confidence, torch.zeros_like(confidence))

        # バッチ対応のメタデータ
        metadata = {
            "winner_id": winner_idx,  # (B,) tensor
            "winner_label": [self.config["labels"][i.item()] for i in winner_idx],  # List of strings
            "confidence": confidence,  # (B,) tensor
            "raw_score": max_raw_score,  # (B,) tensor
            "gate_open": is_gate_open,  # (B,) boolean tensor
            "raw_distribution": w_hat.detach().cpu().numpy(),  # (B, n_probes) numpy array
        }

        return s0, metadata

    def vicara_refinement(self, s0: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray], List[float]]:
        """
        [Refinement Phase] 外部入力を遮断し、再帰ループで状態を純化します。

        Args:
            s0 (torch.Tensor): 初期状態ベクトル。

        Returns:
            s_final (torch.Tensor): 収束後の状態ベクトル。
            trajectory (list): 各ステップの状態ベクトルの履歴。
            energies (list): 各ステップの状態変化量（エネルギー）の履歴。
        """
        s_t = s0.clone()
        trajectory = [s_t.detach().numpy().flatten()]
        energies = []

        for _ in range(self.config["refine_steps"]):
            s_prev = s_t.clone()

            # 純化プロセス: オートエンコーダを通し、ノイズを除去
            residual = self.refiner(s_t)

            # 慣性項付き更新 (Exponential Moving Average)
            # 急激な変化を防ぎ、安定したアトラクタへ誘導する
            s_t = 0.7 * s_t + 0.3 * residual

            # 収束エネルギー (Stability Loss) の計算
            energy = torch.norm(s_t - s_prev).item()
            energies.append(energy)
            trajectory.append(s_t.detach().numpy().flatten())

            # 早期終了判定 (Appanā - Full Absorption)
            if energy < 1e-4:
                break

        return s_t, trajectory, energies

    def compute_dynamics(self, current_log: Dict) -> Optional[Dict]:
        """
        [Cetana Dynamics] 直前の推論結果と比較し、意図の遷移を分析します。

        Args:
            current_log (dict): 現在のステップのVitakkaメタデータ。

        Returns:
            dynamics_log (dict, optional): 遷移タイプ(Sustain/Shift)を含む辞書。初回はNone。
        """
        if not self.history_log:
            return None

        prev_log = self.history_log[-1]["probe_log"]

        # 遷移タイプの判定
        if current_log["winner_id"] == prev_log["winner_id"]:
            trans_type = "Sustain"
        else:
            trans_type = "Shift"

        return {
            "from": prev_log["winner_label"],
            "to": current_log["winner_label"],
            "type": trans_type,
            "confidence_delta": current_log["confidence"] - prev_log["confidence"],
        }

    def forward_step(self, x_input: torch.Tensor, step_idx: int) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        1タイムステップ分の瞑想プロセス（検索→純化→記録）を実行します。

        Args:
            x_input (torch.Tensor): 入力ベクトル。(Shape: [1, dim])
            step_idx (int): 現在のタイムステップ番号。

        Returns:
            Tuple[torch.Tensor, Dict]: 収束した状態と全ログデータのペア。
            ゲートが閉じた場合は None を返します。
        """
        # 1. Vitakka (Search) - x_inputは (1, Dim) なので、vitakka_searchの出力も単一アイテム用として処理
        s0_batch, probe_log_batch = self.vitakka_search(x_input)

        # バッチ出力から単一アイテムのメタデータを抽出
        probe_log = {
            "winner_id": probe_log_batch["winner_id"].item(),
            "winner_label": probe_log_batch["winner_label"][0],
            "confidence": probe_log_batch["confidence"].item(),
            "raw_score": probe_log_batch["raw_score"].item(),
            "gate_open": probe_log_batch["gate_open"].item(),
            "raw_distribution": probe_log_batch["raw_distribution"][0],
        }
        s0 = s0_batch

        if not probe_log["gate_open"]:
            # ノイズとして棄却された場合の処理
            return None

        # 2. Vicāra (Refinement)
        s_final, trajectory, energies = self.vicara_refinement(s0)

        # 3. Sati/Sampajañña (Meta-Cognition)
        dynamics = self.compute_dynamics(probe_log)

        # ログの集約
        full_log = {
            "step": step_idx,
            "probe_log": probe_log,
            "dynamics": dynamics,
            "energies": energies,
            "s_norm": torch.norm(s_final).item(),
        }

        # 履歴に追加
        self.history_log.append(full_log)

        return s_final, full_log

    def forward_sequence(self, x_input_stream: List[torch.Tensor], reset_history: bool = True) -> List[Dict]:
        """
        時系列入力ストリーム全体に対して、連続的な瞑想プロセス（Citta-santāna）を実行します。

        リスト形式で渡された時系列入力を順番に処理し、
        ゲートが開いた（瞑想が成立した）ステップのログのみを抽出して返します。

        Args:
            x_input_stream (List[torch.Tensor]): 時系列順に並んだ入力ベクトルのリスト。
            reset_history (bool, optional): 実行前に過去の履歴（history_log）を消去するかどうか。
                                            新しいセッションを始める場合は True (デフォルト)。
                                            前のセッションから意識の流れを継続する場合は False。

        Returns:
            List[Dict]: ゲートが開いた全ステップのログデータのリスト。
                        ゲートが閉じたステップの情報はここには含まれませんが、
                        コンソール出力や内部状態のスキップ処理としては機能しています。
        """
        if reset_history:
            self.history_log = []

        outputs = []

        for i, x_in in enumerate(x_input_stream):
            # 1ステップ実行
            result = self.forward_step(x_in, step_idx=i)

            # ゲートが開いた場合のみ、結果リストに追加
            if result is not None:
                _, full_log = result
                outputs.append(full_log)

        return outputs
