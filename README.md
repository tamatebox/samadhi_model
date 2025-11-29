# Samadhi Model (Deep Convergence Architecture)

> **"From Generation to Convergence."**
> 生成（Divergence）から、収束（Convergence）へ。

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Samadhi Model**は、従来の「系列予測（Next Token Prediction）」を行う生成AIに対し、対象の「本質的構造の抽出」と「内部状態の不動化」を目的とした、新しい**再帰型アテンション・アーキテクチャ**です。

情報の水平的な拡張（おしゃべりな生成）ではなく、垂直的な深化（静寂な洞察）を工学的に実装します。

---

## 🧘 Concept & Philosophy

現代のLLM（Transformer）は、確率分布の波に乗って次々とトークンを生成する「拡散的」な性質を持ちます。対して **Samadhi Model** は、力学系のアトラクタ（不動点）へ向かって状態を遷移させる「収束的」なモデルです。

仏教心理学における禅定（Samadhi）のプロセスを、以下のエンジニアリング・モジュールとして実装しています。

| Module | Buddhist Term | Engineering Concept | Function |
| :--- | :--- | :--- | :--- |
| **Vitakka** | 尋 (Initial Application) | **Active Probing** | カオス的な入力から「意図（Probe）」を検索・捕捉する。 |
| **Sati** | 正知 (Clear Comprehension) | **Gating Mechanism** | ノイズや幻覚（Hallucination）を検知し、処理を遮断する。 |
| **Vicāra** | 伺 (Sustained Application) | **Recurrent Refinement** | 外部入力を遮断し、再帰ループで状態エネルギーを最小化（純化）する。 |
| **Santāna** | 相続 (Continuity) | **State Dynamics Log** | 意図の遷移（集中・転換・散乱）を時系列で追跡する。 |

---

## 🚀 Key Features

* **Convergence (収束性):** 出力はテキストストリームではなく、エントロピーが極小化された単一の「純化状態（Purified State）」です。
* **O(1) Inference:** 推論コストは入力長（Context Length）に依存せず、収束までのステップ数（定数）のみに依存します。
* **Noise Robustness:** 強力なGating機構により、意味のない入力（雑念）に対しては計算リソースを割かず「沈黙」を返します。
* **Explainability (XAI):** 「なぜその対象に注目したか」「どのように集中が深まったか」がログとして完全に可視化されます。

---

## 📂 Project Structure

```bash
.
├── data/               # MNIST, Waveform datasets
├── docs/               # Theoretical specifications
├── src/
│   ├── components/     # Vitakka (Search) and Vicara (Refinement) modules
│   ├── model/          # Core Architectures (SamadhiCore, ConvSamadhi)
│   └── train/          # Trainer Implementations (Base, Supervised, Unsupervised)
├── test/               # Demos and Training Examples
│   ├── test_minist.py
│   ├── test_trainer_cbsd68.py
│   ├── test_trainer_cifar10.py
│   ├── test_trainer_mnist.py
│   └── test_unsupervised_mnist.py
├── main.py             # Entry point
└── pyproject.toml      # Project configuration (uv)
````

-----

## ⚡ Quick Start

### Prerequisites

このプロジェクトはパッケージマネージャーに `uv` を使用しています。

```bash
# Install dependencies
uv sync
```

### 1\. Basic Usage (Signal Purification)

ノイズ混じりの波形から、特定の信号（意図）を抽出する最小限のデモです。

```python
from src.model import SamadhiCore, CONFIG
import torch

# Initialize Model
CONFIG["dim"] = 64
model = SamadhiCore(CONFIG)

# Input: Noise mixed with a target signal
noisy_input = torch.randn(1, 64)

# Execute One Step of Meditation
s_final, log = model.forward_step(noisy_input, step_idx=0)

if log["probe_log"]["gate_open"]:
    print(f"Focused on: {log['probe_log']['winner_label']}")
    print(f"Converged Energy: {log['energies'][-1]}")
else:
    print("[--- SILENCE ---] Distraction detected.")
```

### 2\. Run Demos

**Visual Samadhi (MNIST Inference Demo)**
ノイズだらけの画像から、モデルが「数字の概念」を見出し、鮮明なイメージへ収束させる過程を可視化します。

```bash
uv run test/test_minist.py
```

**Supervised Training Loop (MNIST Denoising)**
MNISTデータセットを用いた教師あり学習のデモです。ノイズの多い画像から数字の概念を抽出し、純化する過程を学習します。

```bash
uv run test/test_trainer_mnist.py
```

**Unsupervised Training Loop (MNIST Concept Discovery)**
MNISTデータセットを用いた教師なし学習のデモです。モデルが自律的にデータの背後にある概念（プローブ）を学習します。

```bash
uv run test/test_unsupervised_mnist.py
```

-----

## 📊 Architecture Comparison

| Feature | Transformer (GPT) | Samadhi Model (Ours) |
| :--- | :--- | :--- |
| **Vector Flow** | Divergence (発散・生成) | Convergence (収束・純化) |
| **Time Complexity** | $O(N^2)$ (Quadratic) | $O(1)$ (Constant/Iterative) |
| **Dependency** | Context History | Current State Only (Markov) |
| **Objective** | Likelihood Maximization | Stability Energy Minimization |
| **Output** | Probability Distribution | Fixed Point Attractor |

-----

## 🛠 Roadmap

  - [x] **v1.0:** Theoretical Definition (Concept Proof)
  - [x] **v2.2:** Waveform Simulation (Vitakka/Vicāra Implemented)
  - [x] **v2.3:** Gating & Meta-Cognition (Sati Implemented)
  - [ ] **v3.0:** NLP Implementation (Text Summarization/Concept Extraction)
  - [ ] **Future:** Multi-Agent Samadhi (Dialogue of Insight)

-----

## 📜 License

MIT License
