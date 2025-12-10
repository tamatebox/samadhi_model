# Satipatthana Framework (Introspective Deep Convergence Architecture)

> **"From Chaos to Essence, with Awareness."**

![Status](https://img.shields.io/badge/Status-Experimental-orange)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**Satipatthana Framework** is an **introspective recursive attention architecture** designed for **converging to essential structures (Samatha)** from complex, noisy data, and **introspecting (Vipassana)** the process to explain its own confidence.

The name "Satipatthana" (念処) means "establishment of mindfulness", symbolizing the architecture's essence of discerning truth through self-observation and introspection.

Instead of expanding information horizontally (generation), it implements a vertical deepening (insight) that reduces information entropy to reach a stable, meaningful state (Attractor), while maintaining awareness of its own reasoning process.

---

## 🧘 Concept & Philosophy

**Satipatthana** is a three-phase cognitive architecture:

1. **Samatha (Convergent Thinking):** Converges chaotic input to a stable fixed-point through Vitakka (intentional initialization) and Vicara (refinement).
2. **Vipassana (Introspective Self-Awareness):** Monitors the thinking process (SantanaLog) to generate trust scores and context vectors.
3. **Conditional Decoding (Humble Expression):** Safe output that integrates state with "uncertainty awareness".

It implements Buddhist psychology concepts as engineering modules:

| Module | Buddhist Term | Engineering Concept | Function |
| :--- | :--- | :--- | :--- |
| **Vitakka** | 尋 (Initial Application) | **Semantic Initialization** | Generates meaningful initial state $S_0$ from concept probes. |
| **Vicara** | 伺 (Sustained Application) | **Contractive Refinement** | Single-step state update towards fixed point. |
| **Sati** | 正知 (Clear Comprehension) | **Convergence Check** | Monitors state change and determines stopping. |
| **Santāna** | 相続 (Continuity) | **Thinking Trajectory** | Records state history for explainability. |
| **Vipassana** | 観 (Insight) | **Meta-Cognition** | Evaluates reasoning quality and generates confidence. |

> 📖 For detailed architecture specifications, see [Japanese](docs/model.md) / [English](docs/model_en.md).

> 📜 For theoretical foundations, see [Japanese Theory](docs/theory/jp.md) / [English Theory](docs/theory/en.md).

---

## 🚀 Key Features

* **Three-Engine Architecture:** SamathaEngine (convergence) + VipassanaEngine (introspection) + ConditionalDecoder (expression).
* **4-Stage Curriculum Training:** Progressive training (Adapter → Samatha → Vipassana → Decoder) for stable learning.
* **Self-Awareness:** Vipassana provides trust scores (0.0–1.0) indicating confidence in outputs.
* **Modular Framework:** Easily swap Adapters (CNN, LSTM, MLP, Transformer) and components.
* **Type-Safe Configuration:** Robust configuration management using Dataclasses and Enums.
* **Convergence:** Output is not a stream, but a single "Purified State" with minimized entropy.
* **O(1) Inference:** Inference cost depends only on convergence steps (constant), not input length.
* **Explainability (XAI):** Full visualization via SantanaLog of "how thinking evolved".

---

## 🌟 Potential Applications

The unique properties of Satipatthana make it suitable for tasks requiring deep insight, state stability, and self-aware confidence:

1. **Biosignal Analysis (Healthcare):** Extract stable physiological states from noisy EEG or heart rate data, with confidence scores.
2. **Anomaly Detection (Forensics):** Identify "essential anomalies" with uncertainty-aware predictions.
3. **Human Intent Analysis (UX/Psychology):** Capture deep user intent with explainable reasoning trajectories.
4. **Autonomous Agents (Robotics):** Stable decision-making with self-assessment of confidence.
5. **LLM Hallucination Detection:** Use Vipassana to detect "confident lies" in language model outputs.

---

## 📂 Project Structure

```bash
.
├── data/               # Datasets
├── docs/               # Theoretical specifications and plans
├── notebooks/          # Experiments and Analysis (Jupyter)
├── samadhi/
│   ├── configs/        # Configuration System
│   │   ├── system.py   # Root SystemConfig
│   │   ├── factory.py  # Config Factories
│   │   ├── adapters.py # Adapter Configs
│   │   └── ...
│   ├── components/     # Modularized Components
│   │   ├── adapters/   # Input Adapters (MLP, CNN, LSTM, Transformer)
│   │   ├── augmenters/ # Input Augmentation (Identity, Gaussian)
│   │   ├── decoders/   # Output Decoders (Reconstruction, Conditional)
│   │   ├── vitakka/    # Search Modules
│   │   ├── vicara/     # Refinement Modules
│   │   ├── refiners/   # Core refinement networks (MLP, GRU)
│   │   ├── sati/       # Convergence Monitors
│   │   └── vipassana/  # Meta-Cognition Modules
│   ├── core/           # Core Engines (SamathaEngine, VipassanaEngine, SamadhiSystem)
│   ├── train/          # Training Logic
│   │   └── v4_trainer.py  # 4-Stage Curriculum Trainer
│   └── utils/          # Utility functions
├── tests/              # Unit Tests
└── pyproject.toml      # Project configuration (uv)
```

### Logging

The framework utilizes a centralized logging system managed by `samadhi/utils/logger.py`. For consistent logging behavior, see [docs/logging.md](docs/logging.md).

-----

## ⚡ Quick Start

### Prerequisites

This project uses `uv` as its package manager.

```bash
# Install dependencies
uv sync
```

### 1. Basic Usage (Three-Phase Inference)

```python
import torch
from samadhi.core.system import SamadhiSystem
from samadhi.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from samadhi.configs import create_adapter_config, create_vicara_config

# Configure the system
config = SystemConfig(
    samatha=SamathaConfig(
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=64),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32),
    ),
)

# Build and run
system = SamadhiSystem(config)
input_data = torch.randn(1, 784)

# Three-phase inference
result = system(input_data)

print(f"Output shape: {result.output.shape}")
print(f"Converged state shape: {result.s_star.shape}")
print(f"Trust score: {result.trust_score.item():.3f}")  # 0.0-1.0
print(f"Thinking steps: {len(result.santana)}")
```

### 2. Training (4-Stage Curriculum)

Train using the 4-stage curriculum trainer with Hugging Face integration.

```python
from samadhi.train import SamadhiV4Trainer
from samadhi.core.system import SamadhiSystem, TrainingStage
from transformers import TrainingArguments

# Build SamadhiSystem
system = SamadhiSystem(config)

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
)

# Initialize Trainer
trainer = SamadhiV4Trainer(
    model=system,
    args=args,
    train_dataset=dataset,
)

# Run full 4-stage curriculum
results = trainer.run_curriculum(
    stage0_epochs=5,   # Adapter pre-training
    stage1_epochs=10,  # Samatha training (convergence)
    stage2_epochs=5,   # Vipassana training (meta-cognition)
    stage3_epochs=5,   # Decoder fine-tuning
)
```

### 3. Using Trust Scores

```python
# Inference with confidence check
result = system(input_data)

if result.trust_score > 0.8:
    # High confidence - use output directly
    prediction = result.output
else:
    # Low confidence - take safety measures
    print("Warning: Low confidence prediction")
    # Trigger fallback, search, or abstain
```

---

## 📚 Notebook Demos

The `notebooks/` directory contains Jupyter Notebooks demonstrating the framework's capabilities:

* **MNIST Demo (`mnist_demo.ipynb`):** Visualizes the "purification" process of noisy MNIST digits.
* **Fraud Detection Demo (`fraud_unsupervised_detection_explained.ipynb`):** Anomaly detection using unsupervised learning.
* **Time Series Demo (`time_series_anomaly_detection.ipynb`):** Anomaly detection on time series data.

### How to Run

```bash
# Install Jupyter Lab
uv pip install "jupyterlab>=3"

# Start Jupyter Lab
jupyter lab
```

Navigate to `notebooks/` and open any `.ipynb` file.

---

## 🛠 Roadmap

* [x] **v1.0:** Theoretical Definition (Concept Proof)
* [x] **v2.x:** Core Implementation (Vitakka, Vicāra, Sati)
* [x] **v3.0:** Framework Refactoring (Modularization, Builder, HF Trainer)
* [x] **v4.0:** **Introspective Architecture** (Vipassana, SamadhiSystem, 4-Stage Curriculum, Satipatthana naming)
* [ ] **Future:** NLP Integration (LLM Hallucination Detection)
* [ ] **Future:** Multi-Agent Satipatthana (Collaborative Insight)

-----

## 📜 License

MIT License
