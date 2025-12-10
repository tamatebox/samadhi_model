# Satipatthana Framework (Introspective Deep Convergence Architecture) Specification

**Version:** 4.0 (The Three Engines & Guided Convergence)
**Status:** Active Specification

-----

## 1. Concept Definition

The **Satipatthana Framework** is an **introspective recursive attention architecture** designed to **converge towards essential structures (Samatha)** from chaotic information streams and **introspect (Vipassana)** that process to explain its own confidence.

The name "Satipatthana" (念処) means "establishment of mindfulness", symbolizing the architecture's essence of discerning truth through self-observation and introspection.

* **Core Philosophy:** In contrast to traditional "divergence/generation" models, adopts a three-phase approach of "convergence/introspection/expression".
* **Operational Mode:** Progressive knowledge acquisition through a 4-stage curriculum (Adapter → Samatha → Vipassana → Decoder).

-----

## 2. System Architecture

This framework consists of three main engines (Samatha, Vipassana, Decoder) and modular components that compose them.

### 2.1. Data Flow Overview

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

**Role:** World model. Converges any input to a "meaningful point".

**Input:** Raw Data `X` (Batch, *)
**Output:**

* `S*` (Batch, Dim): Converged latent state
* `SantanaLog`: Object recording the thinking trajectory
* `severity` (Batch,): Noise intensity (for Vipassana target)

**Component Structure:**

| Component | Role |
|:---|:---|
| **Adapter** | Projects and normalizes raw input to latent space |
| **Augmenter** | Applies noise/perturbation to input (during training) |
| **Vitakka** | Probe-based initial state $S_0$ generation |
| **Vicara** | Single-step state update ($S_t \rightarrow S_{t+1}$) |
| **Sati** | Convergence check and stopping control |

**Features:** Independent of tasks or labels, performs only "structure extraction". Internal perturbation control is possible via `drunk_mode` flag.

### 2.3. Engine 2: VipassanaEngine (The Observer)

**Role:** Meta-cognition. Monitors whether Samatha's thinking process (log) was sound.

**Input:** `S*` (Batch, Dim) + `SantanaLog`
**Output:**

* `V_ctx` (Batch, context_dim): Hint information for decoder (embedding of "doubt")
* `α` (Batch, 1): Trust score (0.0–1.0)

**Structure:** `StandardVipassana` (LogEncoder + ConfidenceMonitor)

### 2.4. Engine 3: ConditionalDecoder (The Speaker)

**Role:** Expression. Integrates state and context into human-understandable form.

**Input:** `S*` (Batch, Dim) + `V_ctx` (Batch, context_dim) → Concatenate → (Batch, Dim + context_dim)
**Output:** `Y` (Batch, output_dim)

**Features:** Enables "humble expression"—when uncertain, output reflects that uncertainty (e.g., wider variance). **The only Decoder used during inference.**

### 2.5. Reconstruction Heads & AuxHead (Training Auxiliary)

Auxiliary modules for training stabilization. **Not used during inference.**

* **`adapter_recon_head`** (Stage 0): Reconstructs original input from Adapter output `z`
* **`samatha_recon_head`** (Stage 1): Reconstructs original input from converged point `S*`
* **`AuxHead`** (Stage 1): Auxiliary head for task prediction from `S*` (dimension: $d$)

#### Important: Relationship between AuxHead and ConditionalDecoder

| Module | Input Dimension | Purpose | Handling in Stage 3 |
|:---|:---|:---|:---|
| `AuxHead` | $d$ (`S*` only) | Stage 1 Guidance learning | **Discarded** |
| `ConditionalDecoder` | $d + c$ (`S*` ⊕ `V_ctx`) | Inference from Stage 3 onwards | Trained from scratch |

Stage 1's `AuxHead` and Stage 3's `ConditionalDecoder` are **physically separate modules due to different input dimensions**. `AuxHead` weights are not transferred to Stage 3; `ConditionalDecoder` is trained from scratch.

-----

## 3. Component Details

### 3.1. Adapter (Manasikāra - Input Adaptation)

**Function:** Projects and normalizes raw external input $X_{raw}$ to latent space.

* **Interface:** `BaseAdapter`
* **Implementations:** `MlpAdapter`, `CnnAdapter`, `LstmAdapter`, `TransformerAdapter`
* **Output:** Latent vector $z \in \mathbb{R}^d$

### 3.2. Augmenter (Input Perturbation)

**Function:** Applies environmental noise or perturbation to input.

* **Interface:** `BaseAugmenter`
* **Implementations:** `IdentityAugmenter`, `GaussianNoiseAugmenter`
* **Output:** `(x_augmented, severity)` - severity is per-sample noise intensity

### 3.3. Vitakka (Search & Orientation)

**Function:** Initial attractor search in latent space.

1. **Active Resonance:** Calculates resonance between concept probes $\mathbf{P}$ and input
2. **$S_0$ Generation:** Uses winner probe as Query to generate initial state

* **Output:** `(s0, metadata)` - metadata includes winner_id, probs, etc.

### 3.4. Vicara (Single-Step Refinement)

**Function:** Single-step state update.

$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

* **Interface:** `BaseVicara`
* **Implementations:** `StandardVicara`, `WeightedVicara`, `ProbeSpecificVicara`
* **Responsibility:** Single-step update only. Loop control is delegated to SamathaEngine.

**Variants:**

| Class | Description |
|:---|:---|
| `StandardVicara` | State update with single Refiner. Simplest |
| `WeightedVicara` | Weighted combination of multiple Refiners |
| `ProbeSpecificVicara` | Selects Refiner based on Vitakka's winner probe/probability |

### 3.5. Sati (Mindfulness - Convergence Check)

**Function:** Convergence check and stopping control.

* **Interface:** `BaseSati`
* **Implementations:** `FixedStepSati`, `ThresholdSati`
* **Stop Condition:** Stops when state change energy $||S_{t+1} - S_t||$ falls below threshold $\epsilon$

### 3.6. Vipassana (Introspection)

**Function:** Meta-cognition module that monitors Samatha's thinking log and evaluates logical consistency and confidence.

* **Interface:** `BaseVipassana`
* **Implementation:** `StandardVipassana`
* **LogEncoder:** Compresses time-series log $\mathcal{T}$ into fixed-length vector
  * **Recommended Implementation:** Bi-LSTM or Transformer Encoder (1-2 layers). A time-series model is essential to capture "order" of thinking and "acceleration of convergence".
* **ConfidenceMonitor:** Detects "hesitation" or "contradiction", outputs trust score $\alpha$ and context vector $V_{ctx}$

**Fallback Strategy:** When $\alpha < \text{threshold}$ during inference:

* Output default answer ("I don't know")
* Or maximize output distribution variance
* Or trigger search/answer refusal

-----

## 4. Mathematical Formulation

### 4.1. Samatha Phase (Convergence)

**State update rule:**
$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

**Stop condition (Sati):**
$$\text{Stop if } ||S_{t+1} - S_t|| < \epsilon_{sati}$$

### 4.2. Vipassana Phase (Introspection)

Calculates trust from thinking log $\mathcal{T} = [S_0, \dots, S^*]$.

$$V_{ctx} = \text{Encoder}(\mathcal{T})$$
$$\alpha = \sigma(\text{Linear}(V_{ctx})) \in [0, 1]$$

* Target ($\hat{\alpha}$): Clean=1.0, Mismatch/Drunk=0.0

### 4.3. Loss Function (Stage-wise)

Objective function switches per training stage.

* **Stage 0 (Adapter Pre-training):** Reconstruction Only
    $$\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$$

* **Stage 1 (Samatha Training):** Stability + Reconstruction + (Optional) Label Guidance
    $$\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{task}(y, \text{AuxHead}(S^*))$$

* **Stage 2 (Vipassana Training):** Binary Cross Entropy (Contrastive)
    $$\mathcal{L}_2 = \text{BCE}(\alpha, \hat{\alpha})$$

* **Stage 3 (Decoder Fine-tuning):** Task Specific Loss
    $$\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$$

-----

## 5. Data Structures

### 5.1. SantanaLog (Thinking Trajectory)

Object that records state history during the convergence process.

```python
class SantanaLog:
    def add(self, state: Tensor) -> None:
        """Add state to trajectory"""

    def to_tensor(self) -> Tensor:
        """Convert trajectory to tensor (Steps, Batch, Dim)"""

    def __len__(self) -> int:
        """Number of recorded steps"""
```

### 5.2. SystemOutput (Inference Output)

```python
@dataclass
class SystemOutput:
    output: Tensor        # Decoded result
    s_star: Tensor        # Converged latent state
    v_ctx: Tensor         # Vipassana context vector
    trust_score: Tensor   # Trust score (0.0–1.0)
    santana: SantanaLog   # Thinking trajectory
    severity: Tensor      # Noise intensity
```

-----

## 6. Algorithm Flow

### 6.1. Inference Flow

```python
def inference(x: Tensor) -> SystemOutput:
    # Phase 1: Samatha (Convergence)
    s_star, santana, severity = samatha_engine(x, run_augmenter=False)

    # Phase 2: Vipassana (Introspection)
    v_ctx, trust_score = vipassana_engine(s_star, santana)

    # Phase 3: Decode (Expression)
    output = conditional_decoder(concat(s_star, v_ctx))

    return SystemOutput(output, s_star, v_ctx, trust_score, santana, severity)
```

### 6.2. SamathaEngine Internal Flow

```python
def samatha_forward(x, noise_level=0.0, run_augmenter=True):
    # Augment (training only)
    if run_augmenter:
        x_aug, severity = augmenter(x, noise_level)
    else:
        x_aug, severity = x, zeros(batch_size)

    # Adapt
    z = adapter(x_aug)

    # Vitakka: Initial state generation
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

## 7. Training Curriculum (4-Stage)

### 7.1. Training Policy

| Stage | Name | Trainable | Frozen | Objective |
|:---|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter, adapter_recon_head | All others | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, (samatha_recon_head, AuxHead) | Vipassana, TaskDecoder | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | All others | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | TaskDecoder | All others | Task Specific Loss |

### 7.2. Stage 2 Noise Generation Strategy

Three data generation strategies to teach Vipassana meta-cognition:

1. **Environmental Ambiguity (Augmented Path)**
   * Add noise to input data
   * Target: `1.0 - severity`

2. **Internal Dysfunction (Drunk Path)**
   * Perturb SamathaEngine internals (`drunk_mode=True`)
   * Specific implementations: Increase Dropout rate in Vicara, add temporary noise to Refiner weights, disturb Vitakka's temperature parameter, etc.
   * Target: `0.0`

3. **Logical Inconsistency (Mismatch Path)**
   * Shuffle S* and SantanaLog within batch
   * Target: `0.0`

-----

## 8. Hyperparameters

### Model Architecture

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `latent_dim` | $d$ | 64-256 | Latent space dimension |
| `context_dim` | $c$ | 32-128 | Vipassana output dimension |
| `num_probes` | $K$ | 8-32 | Number of Vitakka probes |
| `max_steps` | $T$ | 6-20 | Maximum Vicara steps |

### Training Strategy

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `sati_threshold` | $\epsilon$ | 1e-4 | Convergence threshold |
| `beta` | $\beta$ | 0.3-0.7 | State update inertia parameter |
| `guidance_weight` | $\lambda_g$ | 0.1-0.5 | (Stage 1) Guidance loss strength |
| `recon_weight` | $\lambda_r$ | 0.1-0.3 | Reconstruction loss strength |

-----

## 9. Core Dynamics: Divergence vs. Convergence

The core dynamics of Satipatthana Framework is based on convergence, contrasting with the divergent approach of traditional generative models.

| Feature | Divergent Models | **Convergent Models** |
|:---|:---|:---|
| **Basic Operation** | Sequence Prediction, Generation, Divergence | State Purification, Stabilization, Convergence |
| **Time Dependency** | Dependent on Context History | Dependent only on Current State (Markovian) |
| **Attention** | Self-Attention (Between Elements) | Recursive Attention (Between State-Probe) |
| **Nature of Inference** | Open/Infinite | **Closed/Finite** |
| **Explainability** | Limited | **Extremely High (SantanaLog)** |
| **Philosophical Basis** | Association, Generation, Expansion | **Meditation (Samadhi), Insight, Essence Extraction** |

-----

## 10. Applications & Training Strategies

For supervised tasks, actively use **Stage 1 Guidance (AuxHead)** to optimize Samatha's convergence space for the task.

| Application Task | Stage 1 Strategy | Stage 2 Role | Stage 3 Decoder |
|:---|:---|:---|:---|
| **Supervised Classification** | Guidance (CE Loss) | Hallucination Check | Classifier (Softmax) |
| **Supervised Regression** | Guidance (MSE Loss) | Uncertainty Est. | Regressor (Linear) |
| **Anomaly Detection** | Reconstruction Only | Anomaly Score (final output) | Identity |
| **Structure Discovery** | Stability Only | Boundary Detection | None |

-----

## 11. Integration with Large Language Models (LLMs)

This architecture functions as a countermeasure against LLM "Hallucination".

1. **Thinking Phase:** Converge LLM Hidden States with Samatha to verify context consistency
2. **Introspection Phase:** Vipassana detects "confident lies (Mismatch)"
3. **Expression Phase:** If score is low, take safety measures (search trigger, answer refusal)

-----

## 12. Architectural Extensibility

Components can be freely combined using `SystemConfig` and various `ComponentConfig`.

### 12.1. Task-Specific Customization Example

| Task | Adapter | Augmenter | Vicara | Decoder |
|:---|:---|:---|:---|:---|
| **Time Series Anomaly Detection** | LSTM | Gaussian | Standard | Reconstruction |
| **Image Classification** | CNN | Identity | Standard | Conditional |
| **Dialogue Intent Estimation** | Transformer | Identity | ProbeSpecific | Conditional |
| **Robot Control** | MLP | Gaussian | Weighted | Conditional |

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
