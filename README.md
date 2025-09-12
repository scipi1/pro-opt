# ProT - Process Transformer

## Overview
ProT (Process Transformer) is a specialized transformer-based model for time series forecasting, particularly designed for process data analysis and prediction. It implements a modified transformer architecture with custom embedding layers, attention mechanisms, and encoder-decoder structures tailored for sequential process data.

The model is capable of handling missing values in time series data and provides interpretable attention mechanisms with entropy regularization to understand the relationships between different parts of the process sequence.

## Architecture

### Core Components

#### Embedding System
- **ModularEmbedding**: Flexible embedding system that supports multiple embedding types:
  - **Time2Vec**: Temporal embeddings for time-based features
  - **SinusoidalPosition**: Positional embeddings as used in the original Transformer
  - **nn_embedding**: Standard embedding lookup tables for categorical variables
  - **linear**: Linear transformation embeddings for continuous values
  - **mask**: Mask embeddings for handling missing values
  - **pass**: Identity embeddings for direct value representation

#### Attention Mechanism
- **ScaledDotAttention**: Implementation of the scaled dot-product attention
- **AttentionLayer**: Wrapper for attention with projection layers
- Support for causal masking and missing value handling
- Entropy regularization for attention interpretability

#### Encoder-Decoder Structure
- **Encoder**: Processes input sequences with self-attention
- **Decoder**: Processes target sequences with self-attention and cross-attention to encoder outputs
- Pre-norm transformer architecture with residual connections
- Separate causal masking for encoder and decoder

### Model Flow
1. Input data is embedded using the modular embedding system
2. Encoder processes the embedded input with optional causal masking
3. Decoder processes the target sequence while attending to encoder outputs
4. Final linear layers produce forecasting and reconstruction outputs

## Project Structure

```
pro-opt/
├── config/                 # Configuration files
├── data/                   # Data directory
├── docs/                   # Documentation
│   ├── model_versions.md   # Model version history
│   ├── todo.md            # Project todo list
│   └── references/        # Reference materials
├── experiments/            # Experiment configurations and results
│   └── training/          # Training experiment configs
├── logs/                   # Training and execution logs
├── notebooks/              # Jupyter notebooks for analysis
├── proT/                   # Main source code
│   ├── modules/            # Core model components
│   │   ├── attention.py    # Attention mechanisms
│   │   ├── decoder.py      # Decoder implementation
│   │   ├── embedding.py    # Modular embedding system
│   │   ├── embedding_layers.py # Base embedding layers
│   │   ├── encoder.py      # Encoder implementation
│   │   ├── extra_layers.py # Additional utility layers
│   │   └── utils.py        # Utility functions
│   ├── subroutines/        # Task-specific subroutines
│   ├── baseline/           # Baseline model implementations
│   ├── simulator/          # Data simulation utilities
│   ├── utils/              # General utilities
│   ├── old_/               # Legacy code
│   ├── callbacks.py        # Training callbacks
│   ├── cli.py              # Command-line interface
│   ├── dataloader.py       # Data loading utilities
│   ├── experiment_control.py # Experiment management
│   ├── forecaster.py       # Lightning module wrapper
│   ├── optuna_opt.py       # Hyperparameter optimization
│   ├── predict.py          # Prediction utilities
│   ├── proT_model.py       # Main model definition
│   └── trainer.py          # Training utilities
├── scripts/                # Utility scripts for cluster execution
└── test/                   # Unit tests
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.2+
- PyTorch Lightning 2.0+
- Other dependencies listed in requirements.txt

### Setting Up the Environment

1. Clone the repository:
```bash
git clone https://github.com/scipi1/pro-opt.git
cd pro-opt
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The project provides a command-line interface for training and hyperparameter optimization:

#### Training

```bash
python -m proT.cli train --exp_id <experiment_id> [options]
```

Options:
- `--exp_id`: Experiment folder containing the config file
- `--debug`: Enable debug mode (default: False)
- `--cluster`: Enable cluster mode for distributed training (default: False)
- `--exp_tag`: Tag for model manifest (default: "NA")
- `--scratch_path`: SCRATCH path for cluster execution
- `--resume_checkpoint`: Resume training from checkpoint
- `--plot_pred_check`: Generate prediction plots after training (default: True)
- `--sweep_mode`: Sweep mode, either 'independent' or 'combination' (default: 'combination')

#### Hyperparameter Optimization

```bash
python -m proT.cli paramsopt --exp_id <experiment_id> --mode <mode> [options]
```

Options:
- `--exp_id`: Experiment folder containing the config file
- `--mode`: Select between ['create', 'resume', 'summary']
- `--cluster`: Enable cluster mode (default: False)
- `--study_name`: Name for the Optuna study (default: "NA")
- `--exp_tag`: Tag for model manifest (default: "NA")
- `--scratch_path`: SCRATCH path for cluster execution
- `--study_path`: Path to existing study for resuming

### Configuration

The model is configured using YAML files. A typical configuration includes:

```yaml
model:
  model_object: "proT"
  
  embed_dim:
    enc_val_emb_hidden: 50
    enc_var_emb_hidden: 50
    enc_pos_emb_hidden: 70
    enc_time_emb_hidden: 90
    dec_val_emb_hidden: 130
    dec_var_emb_hidden: 100
    dec_pos_emb_hidden: 170
    dec_time_emb_hidden: 50
  
  kwargs:
    comps_embed_enc: "concat"
    comps_embed_dec: "concat"
    
    # Attention configuration
    enc_attention_type: "ScaledDotProduct"
    dec_self_attention_type: "ScaledDotProduct"
    dec_cross_attention_type: "ScaledDotProduct"
    n_heads: 2
    causal_mask: False
    enc_causal_mask: True
    dec_causal_mask: True
    
    # Architecture configuration
    e_layers: 1
    d_layers: 1
    d_ff: 400
    d_qk: 100
    activation: "gelu"
    norm: "batch"
    use_final_norm: True
    out_dim: 2
    
    # Dropout configuration
    dropout_emb: 0.2
    dropout_data: 0.2
    dropout_attn_out: 0.0
    dropout_ff: 0.0

training:
  optimization: 1
  base_lr: 0.001
  emb_lr: 0.01
  batch_size: 50
  max_epochs: 1000
  loss_fn: "mse"
  k_fold: 5
  seed: 42
  
  # Entropy regularization
  entropy_regularizer: True
  gamma: 0.05

data:
  dataset: "your_dataset_name"
  filename_input: "X.npy"
  filename_target: "Y.npy"
  val_idx: 4
  pos_idx: 3
```

## Key Features

### Advanced Embedding System
- Modular embedding architecture supporting multiple embedding types
- Separate encoder and decoder embeddings with different configurations
- Support for categorical variables, continuous values, temporal features, and missing value handling

### Flexible Training Options
- Multiple optimization strategies (7 different modes)
- Two-phase training with different optimizers and learning rates
- Sparse gradient support for embedding layers
- Entropy regularization for attention interpretability

### Experiment Management
- YAML-based configuration system
- Hyperparameter optimization with Optuna
- Experiment sweeping capabilities
- Checkpoint management and resuming

### Missing Value Handling
- Built-in support for missing values in time series
- Mask-based attention mechanisms
- Reconstruction capabilities alongside forecasting

## Examples

### Basic Training Example

1. Prepare your data in NumPy format (X.npy for input, Y.npy for target)
2. Create a configuration file in the `experiments/training/<exp_id>` directory
3. Run the training command:

```bash
python -m proT.cli train --exp_id your_experiment_id
```

### Hyperparameter Optimization Example

1. Set up your base configuration
2. Create an Optuna study:

```bash
python -m proT.cli paramsopt --exp_id your_experiment_id --mode create --study_name my_study
```

3. Resume optimization:

```bash
python -m proT.cli paramsopt --exp_id your_experiment_id --mode resume --study_name my_study
```

### Using the Model in Code

```python
import torch
import pytorch_lightning as pl
from proT.proT_model import ProT
from proT.forecaster import TransformerForecaster
from proT.dataloader import ProcessDataModule

# Load configuration
config = {...}  # Your model configuration

# Create data module
data_module = ProcessDataModule(
    data_dir="path/to/data",
    input_file="X.npy",
    target_file="Y.npy",
    batch_size=32,
    num_workers=4
)

# Create model
model = TransformerForecaster(config)

# Train model (using PyTorch Lightning)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)

# Make predictions
trainer.predict(model, data_module)
```

## Model Versions

The project maintains version history in `docs/model_versions.md`. Current version is 5.5 with the following key features:
- Training optimization options
- Entropy regularization
- Embedding for given values
- Separate encoder/decoder causal masks
- Optuna integration for hyperparameter optimization

## References

- Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
- Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector Representation of Time." arXiv preprint arXiv:1907.05321.
- Grigsby, J., et al. (2023). "Spacetimeformer: High-dimensional time series forecasting with self-attention." arXiv preprint.
