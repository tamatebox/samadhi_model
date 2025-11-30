# Samadhi Training Strategy Guide

This document outlines flexible training strategies for the **Samadhi Framework**. By leveraging its modular architecture, you can train components (Adapter, Decoder, Refiner) individually or in combinations to achieve stable convergence and high performance.

-----

## 🎯 Modular Training Patterns

Since Samadhi consists of distinct modules, you don't always have to train everything End-to-End from scratch. Step-by-step pre-training often yields better stability.

| Pattern | Components Trained | Objective | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **1. Adapter Only** | `Adapter` | Stability / Contrastive | Initializing latent space without labels. |
| **2. Decoder Only** | `Decoder` | Task Loss (CE/MSE) | When latent space is already fixed/rich (e.g., using pre-trained BERT embeddings). |
| **3. Autoencoder** | `Adapter` + `Decoder` | Reconstruction | Learning feature representation before introducing complex dynamics. |
| **4. Dynamics Only** | `Adapter` + `Refiner` | Stability ($ \|S_{t+1} - S_t\| $) | Learning internal convergence rules (grammar of thought) without specific output targets. |
| **5. Readout** | `Decoder` + `Refiner` | Task Loss | Fine-tuning output mapping from a converged state. |
| **6. Full System** | **All** | **Total Samadhi Loss** | Final optimization. Orchestrates search, purification, and expression. |

-----

## 🚀 Case Studies: Recommended Roadmaps

### Case 1: Anomaly Detection (Unsupervised / Semi-supervised)
**Goal:** Detect outliers by checking if they fail to converge or reconstruct poorly.

1.  **Phase 1: Autoencoder Pre-training (Pattern 3)**
    *   **Train:** `Adapter` + `Decoder`
    *   **Data:** Normal data only.
    *   **Loss:** Reconstruction Loss.
    *   **Why:** Ensure the model can map input $\to$ latent $\to$ input. Establish a baseline capability.

2.  **Phase 2: Full Samadhi Training (Pattern 6)**
    *   **Train:** All (Vitakka/Vicāra active).
    *   **Data:** Normal data only.
    *   **Loss:** Reconstruction + Stability + Entropy.
    *   **Why:** Learn to *purify* noisy normal data into a clean state. Anomalies will resist this purification, resulting in high loss/instability.

### Case 2: Classification with Limited Labels (Few-Shot / Transfer)
**Goal:** High accuracy classification using a small labeled dataset.

1.  **Phase 1: Unsupervised Dynamics Learning (Pattern 4)**
    *   **Train:** `Adapter` + `Refiner` (Decoder detached).
    *   **Data:** Large amount of unlabeled data.
    *   **Loss:** Stability + Sparsity.
    *   **Why:** Learn the inherent structure (manifold) of the data domain. Create strong attractors for common patterns.

2.  **Phase 2: Supervised Fine-tuning (Pattern 2 or 6)**
    *   **Train:** `Decoder` (and optionally fine-tune others).
    *   **Data:** Small labeled dataset.
    *   **Loss:** CrossEntropy Loss.
    *   **Why:** Map the already-stable latent states to class labels. Since the internal structure is robust, it requires fewer samples to learn the boundary.

### Case 3: Complex Reasoning / De-noising
**Goal:** Extract clean intent/signal from highly noisy or ambiguous input.

1.  **Phase 1: Adapter Pre-training (Pattern 1)**
    *   **Train:** `Adapter`
    *   **Loss:** Contrastive Loss (SimCLR style).
    *   **Why:** Ensure similar inputs map to nearby points in the Samadhi Space.

2.  **Phase 2: Dynamics & Reconstruction (Pattern 6)**
    *   **Train:** Full System.
    *   **Loss:** Task Loss + Strong Stability Penalty.
    *   **Why:** Enforce the model to find a *stable* interpretation even for noisy inputs.

-----

## 📝 Implementation Note (Objective-Driven)

In the new framework architecture, these phases are managed by switching the **`Objective`** component injected into the Trainer.

```python
# Example: Phase 1 (Autoencoder)
trainer = SamadhiTrainer(
    model, 
    optimizer, 
    objective=ReconstructionObjective() # Only calculates Recon Loss
)
trainer.train(loader)

# Example: Phase 2 (Full System)
trainer = SamadhiTrainer(
    model, 
    optimizer, 
    objective=AnomalyObjective() # Recon + Stability + Margin
)
trainer.train(loader)
```

