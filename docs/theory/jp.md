# Satipatthana Framework

意味的初期化と収束過程による安定・説明可能・制御可能な表現動力学モデル

**Version:** 4.0
**著者:** 井戸亮太
**日付:** 2025-12-01

---

# 要旨（Abstract）

本論文では、**Satipatthana** と名付けた新しいニューラルアーキテクチャを提案する。
Satipatthana は以下の3段階構造を持つ **内省型 Deep Equilibrium Model（Introspective DEQ）** である。

1. **Samatha — 収束による思考**
   Vitakka（意図的初期化）とVicara（収束精製）による安定な不動点への収束。

2. **Vipassana — 内省による自己認識**
   思考過程（SantanaLog）を監視し、信頼度と文脈ベクトルを生成。

3. **Conditional Decoding — 謙虚な表現**
   状態と「自信のなさ」を統合した安全なアウトプット。

この構造により、Satipatthana は従来の深層モデルに比べて：

* **数学的に保証された安定性**（不動点収束）
* **SantanaLog による高い説明可能性**（内的推論過程の可視化）
* **Vipassana による自己認識能力**（信頼度の定量化）
* **Hard/Soft Attention の切替**による学習性と決定性の両立
* **入力長に依存しない O(1) 思考コスト**
* **エネルギー関数を通じた推論過程の制御性**

といった特徴を有する。

Satipatthana を Transformer、Modern Hopfield Network、Deep Equilibrium Model（DEQ）と比較しつつ、その構造的・数学的特異点を明確にし、新しい「収束型認知モデル」として位置づける。

---

# 1. はじめに

Transformer を中心とする現行の深層学習モデルは極めて高性能である一方、次の課題を持つ。

* 自己回帰による「止まれない推論」
* `O(N²)` に増大する注意計算
* 内部状態の不安定性
* 推論過程の不可視性（ブラックボックス性）
* **自身の回答に対する信頼度を知らない**

**Satipatthana はこれらを根本的に解決するアーキテクチャである。**

Satipatthana（念処）という名称は「気づき（Sati）の確立」を意味し、自己観察・内省を通じて真実を見極めるという本アーキテクチャの本質を象徴している。

---

# 2. アーキテクチャ概要

Satipatthana は以下の3つのエンジンから構成される。

```txt
入力 X
  ↓
[SamathaEngine]
  Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati)
  ↓  S*, SantanaLog
[VipassanaEngine]
  LogEncoder → ConfidenceMonitor
  ↓  V_ctx, α
[ConditionalDecoder]
  ↓  Y
出力
```

---

# 2.1 SamathaEngine: 収束による思考

SamathaEngineは以下のコンポーネントで構成される：

| コンポーネント | 役割 |
|:---|:---|
| **Adapter** | 生入力を潜在空間へ投影 |
| **Augmenter** | 入力にノイズを付与（学習時） |
| **Vitakka** | 意味的な初期状態 $S_0$ を生成 |
| **Vicara** | 1ステップの状態更新 |
| **Sati** | 収束判定・停止制御 |

## 2.1.1 Vitakka: 意図的初期化

入力埋め込み `z` と、プロトタイプベクトル `{P_k}` から初期状態を生成する。

$$
\alpha_k = \mathrm{softmax}\!\left( \frac{\langle z, P_k \rangle}{\tau} \right)
$$

$$
S_0 = \sum_k \alpha_k\, P_k
$$

* **学習時:** Soft Attention（τ 大）
* **推論時:** Hard Attention（τ → 0）

Transformer と異なり、**意味的に整合した初期状態から思考が始まる**点が特徴である。

## 2.1.2 Vicara: 1ステップの状態更新

Vicara は縮小写像の形をとる：

$$
S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)
$$

Vicaraは**単一ステップの更新のみ**を担当し、ループ制御はSamathaEngineに委譲される。

**バリエーション:**

| クラス | 説明 |
|:---|:---|
| `StandardVicara` | 単一Refinerで状態更新 |
| `WeightedVicara` | 複数Refinerの重み付け合成 |
| `ProbeSpecificVicara` | Vitakkaの勝者Probeに基づきRefinerを選択 |

## 2.1.3 Sati: 停止ゲート

収束判定：

$$
\Delta_t = \| S_t - S_{t-1} \|
$$

$$
\Delta_t < \epsilon \quad \Rightarrow \quad \text{停止}
$$

Transformer のように「確率的にトークンを生成し続ける」構造ではなく、
Satipatthana は **収束＝停止** する。

## 2.1.4 SantanaLog: 思考軌跡の記録

収束過程の状態履歴を記録するオブジェクト。

$$
\mathcal{S} = \{ S_0, S_1, \dots, S^{*} \}
$$

SantanaLog により：

* 仮説がどのように変化したか
* 確信度がどのように強まったか
* どのステップが重要だったか

が追跡可能となる。注意重みのヒートマップでは得られない、**真正な内部状態の説明性**がある。

---

# 2.2 VipassanaEngine: 内省による自己認識

**メタ認知を担う新しいエンジン。**

VipassanaEngineは、Samathaの思考プロセス（SantanaLog）が健全だったかを監視するメタ認知モジュールである。

**入力:** $S^*$ + $\mathcal{S}$ (SantanaLog)
**出力:**

* $V_{ctx}$: 文脈ベクトル（「迷い」の埋め込み表現）
* $\alpha$: 信頼度スコア (0.0〜1.0)

$$
V_{ctx} = \text{LogEncoder}(\mathcal{S})
$$

$$
\alpha = \sigma(\text{Linear}(V_{ctx})) \in [0, 1]
$$

**学習時のターゲット:**

* Clean（ノイズなし）: $\hat{\alpha} = 1.0$
* Mismatch/Drunk: $\hat{\alpha} = 0.0$

---

# 2.3 ConditionalDecoder: 謙虚な表現

**入力:** $S^* \oplus V_{ctx}$（結合）
**出力:** 最終出力 $Y$

「自信がない時は、自信がないような出力（分散を広げる等）」が可能になり、**謙虚な表現**を実現する。

**推論時に使用される唯一のDecoder**。学習補助用のReconstruction Headは推論時には使用されない。

---

# 3. 数学的基盤

## 3.1 収束性の理論

**縮小写像の理想:**
数学的には、Vicara の写像 $\Phi$ がリプシッツ連続であり、かつそのリプシッツ定数 $c$ が $0 < c < 1$ を満たす場合、バナッハの不動点定理により、任意の初期状態 $S_0$ から唯一の不動点 $S^*$ への収束が保証される。

$$
\|F_\theta(s_a) - F_\theta(s_b)\| \le c\, \|s_a - s_b\|
\quad (0 < c < 1)
$$

## 3.2 実装上のアプローチ

厳密な制約の代わりに、以下の2つのアプローチで「実効的な収束」を担保する。

**1. Dynamics Learning（安定性損失による誘導）**

$$
\mathcal{L}_{stability} = \| S_{t} - S_{t-1} \|^2
$$

これにより、モデルは発散的な挙動に対してペナルティを受け、学習が進むにつれて自然と縮小写像的なダイナミクスを獲得する。

**2. Inertial Update（慣性項による減衰）**

$$
S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)
$$

これにより、内部の写像 $\Phi$ が局所的に不安定な挙動を示したとしても、システム全体としてのリプシッツ定数を引き下げ、安定点へと軟着陸させる。

## 3.3 リアプノフエネルギーの定義

$$
E(s) = \| s - F_\theta(s) \|^2
$$

Vicāra の反復は事実上、

$$
s_t \approx \arg\min_s E(s)
$$

というエネルギー最小化に相当する。

Satipatthana は同時に：

* **暗黙関数モデル（implicit model）**
* **エネルギーベースモデル（EBM）**

としての性質を持つ。

---

# 4. 学習カリキュラム (4-Stage Curriculum)

各コンポーネントを段階的に安定させるための4つのステージで構成される。

| Stage | Name | Train対象 | 目的関数 |
|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | TaskDecoder | Task Specific Loss |

## 4.1 Stage 2: ノイズ生成戦略

Vipassanaにメタ認知能力を習得させるための3種類のデータ生成戦略:

1. **Environmental Ambiguity (Augmented Path)**
   * 入力データへのノイズ付与
   * Target: `1.0 - severity`

2. **Internal Dysfunction (Drunk Path)**
   * SamathaEngine内部の摂動（`drunk_mode=True`）
   * Target: `0.0`

3. **Logical Inconsistency (Mismatch Path)**
   * バッチ内でS*とSantanaLogをシャッフル
   * Target: `0.0`

---

# 5. 既存モデルとの比較

## 5.1 Transformer との比較

| 性質 | Transformer | Satipatthana |
|:---|:---|:---|
| 推論 | 自己回帰・無限シーケンス | 固定点収束・有限 |
| 計算量 | O(N²) | O(N) |
| 安定性 | 担保なし | 数学的に保証 |
| 説明可能性 | 低 | 高（SantanaLog） |
| 初期化 | なし | 意味的初期化 |
| **自己認識** | **なし** | **Vipassana** |

## 5.2 Deep Equilibrium Model（DEQ）との比較

| 観点 | DEQ | Satipatthana |
|:---|:---|:---|
| 初期状態 | ゼロ or ランダム | Vitakka（意味的） |
| 収束 | ○ | ○ |
| 説明可能性 | × | ○（SantanaLog） |
| 注意機構 | なし | 任意に導入可能 |
| **メタ認知** | **×** | **○（Vipassana）** |

## 5.3 Modern Hopfield Network との比較

| 観点 | Hopfield | Satipatthana |
|:---|:---|:---|
| 記憶形式 | 連想記憶 | 意味的アトラクタ |
| エネルギー | 明示的 | 暗黙的 |
| 不動点 | ○ | ○ |
| 説明性 | 中 | 高 |

Satipatthana は**DEQ の一般性**と **Hopfield 的安定性**を統合した位置付けとなる。

---

# 6. 学習戦略

## 6.1 Hard / Soft Attention の切り替え

| フェーズ | Attention | 意図 |
|:---|:---|:---|
| 学習 | Soft | 勾配を流すため |
| 推論 | Hard | 決定論的状態の獲得 |

これにより、「離散的な概念選択」と「微分可能な学習」を両立させる。

---

# 7. 応用分野

* 安定した分類（医療・金融など）
* マルチモーダル表現の融合
* ノイズ除去（音声／生体信号）
* 状態推定（ロボティクス・自律エージェント）
* セーフティクリティカル分野（停止能力）
* 世界モデル（state stabilization）
* **LLMの幻覚検知**（Vipassanaによる内省）

「**安定点に落ち着く**」という特性は極めて広い分野で有用である。

---

# 8. 結論

Satipatthana は以下の要素を統合する。

* 意味的初期化（Vitakka）
* 収束精製（Vicara）
* 停止ゲート（Sati）
* 推論過程の記録（SantanaLog）
* **内省と信頼度推定（Vipassana）**
* **謙虚な表現（ConditionalDecoder）**
* O(1) 思考コスト

Satipatthana は **内省型 DEQ（Introspective DEQ）** という新しいモデルファミリーを形成する。

単なるニューラルネットワークではなく、
**安定・可解釈・制御可能・自己認識可能な「収束型認知モデル」**としての新しいパラダイムである。

---

# 付録A. 擬似コード

```python
def satipatthana_inference(x, samatha, vipassana, decoder):
    # Phase 1: Samatha (収束)
    s_star, santana, severity = samatha(x, run_augmenter=False)

    # Phase 2: Vipassana (内省)
    v_ctx, trust_score = vipassana(s_star, santana)

    # Phase 3: Conditional Decoding (表現)
    output = decoder(concat(s_star, v_ctx))

    return output, trust_score, santana


def samatha_forward(x, adapter, augmenter, vitakka, vicara, sati, max_steps):
    # Augment (学習時のみ)
    x_aug, severity = augmenter(x)

    # Adapt
    z = adapter(x_aug)

    # Vitakka: 意味的初期化
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

---
