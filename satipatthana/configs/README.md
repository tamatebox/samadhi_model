# Configuration System for Satipatthana Framework (v4.0)

This directory contains the 2-Layer Configuration System for the Satipatthana Framework.

## Overview

The v4.0 configuration system uses a 2-layer architecture:

1. **User-Facing Layer** (Recommended): `SatipatthanaConfig`, `CurriculumConfig`, `create_system()`
2. **Internal Layer** (Power Users): `SystemConfig` and component-specific configs

This design solves "Config Hell" - too many nested configs with redundant parameters.

## Quick Start (Recommended)

```python
from satipatthana import SatipatthanaConfig, create_system, CurriculumConfig

# Option 1: From preset string (simplest)
system = create_system("mlp", input_dim=128, output_dim=10)

# Option 2: From config (customizable)
config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    latent_dim=64,
    adapter="mlp",
    n_probes=10,
)
system = config.build()

# Training with CurriculumConfig
from satipatthana.train import SatipatthanaTrainer
trainer.run_curriculum(CurriculumConfig())
```

## Architecture

```text
User-Facing (Simple)                    Internal (Type-Safe)
├── SatipatthanaConfig  ──build()──►   ├── SystemConfig
├── CurriculumConfig                    ├── SamathaConfig
│   ├── StageConfig (0-3)               ├── AdapterConfigs
│   └── NoisePathRatios                 └── ...
└── create_system()  ─────────────►    SatipatthanaSystem
```

## Directory Structure

### User-Facing Configs (New in v4.0)

- `config.py`: `SatipatthanaConfig` - simplified user-facing config
- `curriculum.py`: `CurriculumConfig`, `StageConfig`, `NoisePathRatios` - training settings
- `factory.py`: `create_system()` - factory function for system creation

### Internal Configs (Power Users)

- `system.py`: `SystemConfig`, `SamathaConfig`, `VipassanaEngineConfig` - root configs
- `adapters.py`: `MlpAdapterConfig`, `CnnAdapterConfig`, `LstmAdapterConfig`, `TransformerAdapterConfig`
- `vitakka.py`: `StandardVitakkaConfig`
- `vicara.py`: `StandardVicaraConfig`, `WeightedVicaraConfig`, `ProbeVicaraConfig`
- `sati.py`: `FixedStepSatiConfig`, `ThresholdSatiConfig`
- `vipassana.py`: `StandardVipassanaConfig`, `LSTMVipassanaConfig`
- `augmenter.py`: `IdentityAugmenterConfig`, `GaussianNoiseAugmenterConfig`
- `decoders.py`: `ReconstructionDecoderConfig`, `ConditionalDecoderConfig`, etc.

### Supporting Files

- `base.py`: `BaseConfig` - abstract base class
- `enums.py`: `AdapterType`, `VicaraType`, `DecoderType` enums
- `__init__.py`: Central export point

## Usage Examples

### 1. Simple System Creation

```python
from satipatthana import create_system

# MLP adapter
system = create_system("mlp", input_dim=128, output_dim=10)

# CNN adapter (for images)
system = create_system("cnn", input_dim=784, output_dim=10, img_size=28, channels=1)

# LSTM adapter (for sequences)
system = create_system("lstm", input_dim=32, output_dim=10, seq_len=50)
```

### 2. Customized System with SatipatthanaConfig

```python
from satipatthana import SatipatthanaConfig

config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    latent_dim=64,
    adapter="mlp",
    vicara="standard",
    sati="threshold",
    n_probes=10,
    max_steps=10,
    use_label_guidance=False,
)
system = config.build()
```

### 3. Training with CurriculumConfig

```python
from satipatthana import CurriculumConfig, StageConfig, NoisePathRatios
from satipatthana.train import SatipatthanaTrainer

curriculum = CurriculumConfig(
    stage0=StageConfig(epochs=5, learning_rate=1e-3),
    stage1=StageConfig(epochs=10, learning_rate=5e-4, stability_weight=0.2),
    stage2=StageConfig(epochs=5, learning_rate=1e-4),
    stage3=StageConfig(epochs=5, learning_rate=1e-4),
    noise_path_ratios=NoisePathRatios(
        clean=0.2, augmented=0.2, drunk=0.2, mismatch=0.2, void=0.2
    ),
)

trainer.run_curriculum(curriculum)
```

### 4. Power Users: Direct SystemConfig (Advanced)

```python
from satipatthana.configs import (
    SystemConfig,
    SamathaConfig,
    VipassanaEngineConfig,
    MlpAdapterConfig,
    StandardVitakkaConfig,
    StandardVicaraConfig,
    ThresholdSatiConfig,
    StandardVipassanaConfig,
    ConditionalDecoderConfig,
    ReconstructionDecoderConfig,
)

config = SystemConfig(
    samatha=SamathaConfig(
        dim=64,
        adapter=MlpAdapterConfig(input_dim=128, dim=64),
        vitakka=StandardVitakkaConfig(dim=64, num_probes=10),
        vicara=StandardVicaraConfig(dim=64),
        sati=ThresholdSatiConfig(threshold=1e-4),
        max_steps=10,
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(latent_dim=64, context_dim=32),
    ),
    task_decoder=ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10),
    adapter_recon_head=ReconstructionDecoderConfig(dim=64, input_dim=128),
    samatha_recon_head=ReconstructionDecoderConfig(dim=64, input_dim=128),
)

from satipatthana.core.system import SatipatthanaSystem
system = SatipatthanaSystem(config)
```

## Key Principles

- **Type Safety**: All configuration parameters are defined within Python dataclasses with explicit type hints.
- **2-Layer Design**: Simple user-facing API + detailed internal configs for power users.
- **Dimension Propagation**: `SatipatthanaConfig.build()` automatically propagates `latent_dim` to all components.
- **Sensible Defaults**: All optional parameters have reasonable defaults.
- **Validation**: Each config implements `__post_init__` for validation.

## Extending Configurations

To add a new component type:

1. Add enum member in `enums.py` (if needed)
2. Create `@dataclass` in the relevant file inheriting from `Base*Config`
3. Update factory in `factory.py` (if needed)
4. Export from `__init__.py`

## Related Documentation

- [docs/workflow_guide.md](../../docs/workflow_guide.md) - Implementation cookbook
- [docs/config_summary.md](../../docs/config_summary.md) - Parameter reference
- [docs/training_strategy.md](../../docs/training_strategy.md) - 4-stage training strategy
