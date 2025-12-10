# Satipatthana Framework (Introspective Deep Convergence Architecture) Specification

**Version:** 4.0 (The Three Engines & Guided Convergence)
**Status:** Active Specification

-----

## 1. 概念定義 (Concept Definition)

**Satipatthana Framework**は、カオス的な情報ストリームから対象の『本質的構造への収束（Samatha）』を行い、その過程を『内省（Vipassana）』することで、自身の信頼度を説明できる**内省型・再帰的アテンション・アーキテクチャ**である。

Satipatthana（念処）という名称は「気づき（Sati）の確立」を意味し、自己観察・内省を通じて真実を見極めるという本アーキテクチャの本質を象徴している。

* **Core Philosophy:** 従来の「発散・生成」モデルに対し、「収束・内省・表現」の3段階アプローチを採る。
* **Operational Mode:** 4段階のカリキュラム（Adapter -> Samatha -> Vipassana -> Decoder）による段階的な知性獲得。

-----

## 2. システムアーキテクチャ (System Architecture)

本フレームワークは、3つの主要エンジン（Samatha, Vipassana, Decoder）と、それらを構成するモジュラーコンポーネント群によって構成される。

### 2.1. データフロー概要

```txt
Raw Input (X)
    ↓
[SamathaEngine]
    Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati) → S*, SantanaLog
    ↓
[VipassanaEngine]
    S* + SantanaLog → V_ctx, α (trust_score)
    ↓
[ConditionalDecoder]
    S* + V_ctx → Output (Y)
```

### 2.2. Engine 1: SamathaEngine (The Meditator)

**役割:** 世界モデル。いかなる入力も「意味のある点」に収束させる。

**入力:** Raw Data `X` (Batch, *)
**出力:**

* `S*` (Batch, Dim): 収束した潜在状態
* `SantanaLog`: 思考軌跡を記録したオブジェクト
* `severity` (Batch,): ノイズ強度（Vipassanaターゲット用）

**構成コンポーネント:**

| コンポーネント | 役割 |
|:---|:---|
| **Adapter** | 生入力を潜在空間へ投影・正規化 |
| **Augmenter** | 入力にノイズ/摂動を付与（学習時） |
| **Vitakka** | プローブベースの初期状態 $S_0$ 生成 |
| **Vicara** | 1ステップの状態更新 ($S_t \rightarrow S_{t+1}$) |
| **Sati** | 収束判定・停止制御 |

**特徴:** タスクやラベルには依存せず、「構造の抽出」のみを行う。`drunk_mode` フラグにより内部的な摂動制御が可能。

### 2.3. Engine 2: VipassanaEngine (The Observer)

**役割:** メタ認知。Samathaの思考プロセス（ログ）が健全だったか監視する。

**入力:** `S*` (Batch, Dim) + `SantanaLog`
**出力:**

* `V_ctx` (Batch, context_dim): デコーダーへのヒント情報（「迷い」の埋め込み表現）
* `α` (Batch, 1): 信頼度スコア (0.0〜1.0)

**構成:** `StandardVipassana` (LogEncoder + ConfidenceMonitor)

### 2.4. Engine 3: ConditionalDecoder (The Speaker)

**役割:** 表現。状態と文脈を統合して、人間にわかる形にする。

**入力:** `S*` (Batch, Dim) + `V_ctx` (Batch, context_dim) → Concatenate → (Batch, Dim + context_dim)
**出力:** `Y` (Batch, output_dim)

**特徴:** 「自信がない時は、自信がないような出力（分散を広げる等）」が可能になり、**謙虚な表現**を実現する。**推論時に使用される唯一のDecoder**。

### 2.5. Reconstruction Heads & AuxHead (学習補助)

学習の安定化を目的とした補助モジュール。**推論時には使用されない。**

* **`adapter_recon_head`** (Stage 0用): Adapterの出力 `z` から元入力を再構成
* **`samatha_recon_head`** (Stage 1用): 収束点 `S*` から元入力を再構成
* **`AuxHead`** (Stage 1用): `S*` (次元: $d$) からタスク予測を行う補助ヘッド

**重要: AuxHead と ConditionalDecoder の関係**

| モジュール | 入力次元 | 用途 | Stage 3での扱い |
|:---|:---|:---|:---|
| `AuxHead` | $d$ (`S*`のみ) | Stage 1のGuidance学習 | **破棄** |
| `ConditionalDecoder` | $d + c$ (`S*` ⊕ `V_ctx`) | Stage 3以降の推論 | 新規学習 |

Stage 1の `AuxHead` と Stage 3の `ConditionalDecoder` は**入力次元が異なるため、物理的に別モジュール**である。`AuxHead` の重みは Stage 3 には転移されず、`ConditionalDecoder` はゼロから学習される。

-----

## 3. コンポーネント詳細 (Component Details)

### 3.1. Adapter (Manasikāra - Input Adaptation)

**機能:** 生の外部入力 $X_{raw}$ を潜在空間へ投影・正規化する。

* **Interface:** `BaseAdapter`
* **実装:** `MlpAdapter`, `CnnAdapter`, `LstmAdapter`, `TransformerAdapter`
* **Output:** 潜在ベクトル $z \in \mathbb{R}^d$

### 3.2. Augmenter (Input Perturbation)

**機能:** 入力に対して環境ノイズや摂動を加える。

* **Interface:** `BaseAugmenter`
* **実装:** `IdentityAugmenter`, `GaussianNoiseAugmenter`
* **Output:** `(x_augmented, severity)` - severityはサンプルごとのノイズ強度

### 3.3. Vitakka (Search & Orientation)

**機能:** 潜在空間内での初期アトラクタ探索。

1. **Active Resonance:** 概念プローブ群 $\mathbf{P}$ と入力の共鳴度を計算
2. **$S_0$ Generation:** 勝者プローブをQueryとして初期状態を生成

* **Output:** `(s0, metadata)` - metadataにはwinner_id, probs等を含む

### 3.4. Vicara (Single-Step Refinement)

**機能:** 1ステップの状態更新。

$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

* **Interface:** `BaseVicara`
* **実装:** `StandardVicara`, `WeightedVicara`, `ProbeSpecificVicara`
* **責務:** 単一ステップの更新のみ。ループ制御はSamathaEngineに委譲。

**バリエーション:**

| クラス | 説明 |
|:---|:---|
| `StandardVicara` | 単一Refinerで状態更新。最もシンプル |
| `WeightedVicara` | 複数Refinerの重み付け合成 |
| `ProbeSpecificVicara` | Vitakkaの勝者Probe/確率に基づきRefinerを選択 |

### 3.5. Sati (Mindfulness - Convergence Check)

**機能:** 収束判定と停止制御。

* **Interface:** `BaseSati`
* **実装:** `FixedStepSati`, `ThresholdSati`
* **Stop Condition:** 状態変化エネルギー $||S_{t+1} - S_t||$ が閾値 $\epsilon$ を下回った時点で停止

### 3.6. Vipassana (Introspection)

**機能:** Samathaの思考ログを監視し、論理的整合性と信頼度を評価するメタ認知モジュール。

* **Interface:** `BaseVipassana`
* **実装:** `StandardVipassana`
* **LogEncoder:** 時系列ログ $\mathcal{T}$ を固定長ベクトルに圧縮
  * **推奨実装:** Bi-LSTM または Transformer Encoder (1-2 layers)。思考の「順序」と「収束の加速度」を捉えるには時系列モデルが必須。
* **ConfidenceMonitor:** 「迷い」や「矛盾」を検知し、信頼度スコア $\alpha$ と文脈ベクトル $V_{ctx}$ を出力

**フォールバック戦略:** 推論時に $\alpha < \text{threshold}$ の場合：

* デフォルト回答（"I don't know"）を出力
* または出力分布の分散（Variance）を最大化
* または検索トリガー/回答拒否を発動

-----

## 4. 数理モデル (Mathematical Formulation)

### 4.1. Samatha Phase (Convergence)

**状態更新則:**
$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

**停止条件 (Sati):**
$$\text{Stop if } ||S_{t+1} - S_t|| < \epsilon_{sati}$$

### 4.2. Vipassana Phase (Introspection)

思考ログ $\mathcal{T} = [S_0, \dots, S^*]$ から信頼度を算出する。

$$V_{ctx} = \text{Encoder}(\mathcal{T})$$
$$\alpha = \sigma(\text{Linear}(V_{ctx})) \in [0, 1]$$

* Target ($\hat{\alpha}$): Clean=1.0, Mismatch/Drunk=0.0

### 4.3. Loss Function (Stage-wise)

学習ステージごとに目的関数が切り替わる。

* **Stage 0 (Adapter Pre-training):** Reconstruction Only
    $$\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$$

* **Stage 1 (Samatha Training):** Stability + Reconstruction + (Optional) Label Guidance
    $$\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{task}(y, \text{AuxHead}(S^*))$$

* **Stage 2 (Vipassana Training):** Binary Cross Entropy (Contrastive)
    $$\mathcal{L}_2 = \text{BCE}(\alpha, \hat{\alpha})$$

* **Stage 3 (Decoder Fine-tuning):** Task Specific Loss
    $$\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$$

-----

## 5. データ構造仕様 (Data Structures)

### 5.1. SantanaLog (思考軌跡)

収束過程の状態履歴を記録するオブジェクト。

```python
class SantanaLog:
    def add(self, state: Tensor) -> None:
        """状態を軌跡に追加"""

    def to_tensor(self) -> Tensor:
        """軌跡をテンソル化 (Steps, Batch, Dim)"""

    def __len__(self) -> int:
        """記録されたステップ数"""
```

### 5.2. SystemOutput (推論出力)

```python
@dataclass
class SystemOutput:
    output: Tensor        # デコード結果
    s_star: Tensor        # 収束した潜在状態
    v_ctx: Tensor         # Vipassanaの文脈ベクトル
    trust_score: Tensor   # 信頼度スコア (0.0〜1.0)
    santana: SantanaLog   # 思考軌跡
    severity: Tensor      # ノイズ強度
```

-----

## 6. 処理フロー (Algorithm Flow)

### 6.1. 推論フロー (Inference)

```python
def inference(x: Tensor) -> SystemOutput:
    # Phase 1: Samatha (収束)
    s_star, santana, severity = samatha_engine(x, run_augmenter=False)

    # Phase 2: Vipassana (内省)
    v_ctx, trust_score = vipassana_engine(s_star, santana)

    # Phase 3: Decode (表現)
    output = conditional_decoder(concat(s_star, v_ctx))

    return SystemOutput(output, s_star, v_ctx, trust_score, santana, severity)
```

### 6.2. SamathaEngine内部フロー

```python
def samatha_forward(x, noise_level=0.0, run_augmenter=True):
    # Augment (学習時のみ)
    if run_augmenter:
        x_aug, severity = augmenter(x, noise_level)
    else:
        x_aug, severity = x, zeros(batch_size)

    # Adapt
    z = adapter(x_aug)

    # Vitakka: 初期状態生成
    s0, metadata = vitakka(z)

    # Vicara loop with Sati
    santana = SantanaLog()
    s_t = s0
    santana.add(s_t)

    for step in range(max_steps):
        s_t = vicara(s_t, context=metadata)
        santana.add(s_t)

        should_stop, _ = sati(s_t, santana)
        if should_stop:
            break

    return s_t, santana, severity
```

-----

## 7. 学習カリキュラム (4-Stage Curriculum)

### 7.1. 学習ポリシー

| Stage | Name | Train対象 | Freeze対象 | 目的関数 |
|:---|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter, adapter_recon_head | 他すべて | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, (samatha_recon_head, AuxHead) | Vipassana, TaskDecoder | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | 他すべて | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | TaskDecoder | 他すべて | Task Specific Loss |

### 7.2. Stage 2 ノイズ生成戦略

Vipassanaにメタ認知能力を習得させるための3種類のデータ生成戦略:

1. **Environmental Ambiguity (Augmented Path)**
   * 入力データへのノイズ付与
   * Target: `1.0 - severity`

2. **Internal Dysfunction (Drunk Path)**
   * SamathaEngine内部の摂動（`drunk_mode=True`）
   * 具体的実装: Vicara内のDropout率を上げる、Refinerの重みに一時的ノイズを加算、Vitakkaの温度パラメータを乱す等
   * Target: `0.0`

3. **Logical Inconsistency (Mismatch Path)**
   * バッチ内でS*とSantanaLogをシャッフル
   * Target: `0.0`

-----

## 8. パラメータ設定推奨値 (Hyperparameters)

### Model Architecture

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `latent_dim` | $d$ | 64-256 | 潜在空間の次元 |
| `context_dim` | $c$ | 32-128 | Vipassana出力の次元 |
| `num_probes` | $K$ | 8-32 | Vitakkaのプローブ数 |
| `max_steps` | $T$ | 6-20 | Vicaraの最大ステップ数 |

### Training Strategy

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `sati_threshold` | $\epsilon$ | 1e-4 | 収束判定閾値 |
| `beta` | $\beta$ | 0.3-0.7 | 状態更新の慣性パラメータ |
| `guidance_weight` | $\lambda_g$ | 0.1-0.5 | (Stage 1) Guidance Lossの強さ |
| `recon_weight` | $\lambda_r$ | 0.1-0.3 | Reconstruction Lossの強さ |

-----

## 9. 基本動作の対比 (Core Dynamics)

Satipatthana Frameworkの基本動作は、従来の生成モデルがとる発散的なアプローチとは対照的に、収束を基盤としている。

| 特徴 | 発散モデル (Divergent) | **収束モデル (Convergent)** |
|:---|:---|:---|
| **基本動作** | 系列予測、生成、発散 | 状態の純化、不動化、収束 |
| **時間依存性** | 文脈履歴に依存 | 現在の状態のみに依存 (マルコフ性) |
| **アテンション** | Self-Attention (要素間) | Recursive Attention (状態-プローブ間) |
| **推論の性質** | 開放的・無限 | **閉鎖的・有限** |
| **説明可能性** | 限定的 | **極めて高い (SantanaLog)** |
| **哲学的基盤** | 連想、生成、拡大 | **禅定、洞察、本質抽出** |

-----

## 10. 応用と学習戦略 (Applications & Training Strategies)

教師ありタスクにおいては **Stage 1 Guidance (AuxHead)** を積極的に使用し、Samathaの収束空間をタスク向けに最適化する。

| 応用タスク | Stage 1 Strategy | Stage 2 Role | Stage 3 Decoder |
|:---|:---|:---|:---|
| **教師あり分類** | Guidance (CE Loss) | Hallucination Check | Classifier (Softmax) |
| **教師あり回帰** | Guidance (MSE Loss) | Uncertainty Est. | Regressor (Linear) |
| **異常検知** | Reconstruction Only | Anomaly Score (最終出力) | Identity |
| **構造発見** | Stability Only | Boundary Detection | None |

-----

## 11. 大規模言語モデル (LLM) との連携 (Integration with LLMs)

本アーキテクチャは、LLMの「幻覚（Hallucination）」対策として機能する。

1. **Thinking Phase:** LLMのHidden StatesをSamathaで収束させ、文脈の一貫性を確認
2. **Introspection Phase:** Vipassanaが「自信満々の嘘（Mismatch）」を検知
3. **Expression Phase:** スコアが低い場合、安全策（検索トリガー、回答拒否）を講じる

-----

## 12. アーキテクチャの拡張性 (Architectural Extensibility)

`SystemConfig` と各種 `ComponentConfig` を使用して、コンポーネントを自由に組み合わせることができる。

### 12.1. Task-Specific Customization Example

| タスク | Adapter | Augmenter | Vicara | Decoder |
|:---|:---|:---|:---|:---|
| **時系列異常検知** | LSTM | Gaussian | Standard | Reconstruction |
| **画像分類** | CNN | Identity | Standard | Conditional |
| **対話意図推定** | Transformer | Identity | ProbeSpecific | Conditional |
| **ロボット制御** | MLP | Gaussian | Weighted | Conditional |

### 12.2. Config Example

```python
from satipatthana.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from satipatthana.configs import create_adapter_config, create_vicara_config

config = SystemConfig(
    samatha=SamathaConfig(
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32),
    ),
    use_label_guidance=True,
)
```
