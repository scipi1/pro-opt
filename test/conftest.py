"""
Pytest configuration and shared fixtures for proT tests.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def torch_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture
def basic_config():
    """Basic configuration parameters for tests."""
    return {
        'batch_size': 2,
        'seq_len': 10,
        'd_model': 12,
        'time_comp': 6,
        'd_queries_keys': 8,
        'd_values': 8,
        'n_heads': 3,
        'dropout_qkv': 0.1,
        'dropout_attn_out': 0.1,
        'activation': 'relu',
        'norm': 'layer',
        'use_final_norm': True,
        'dropout_emb': 0.1,
        'e_layers': 2,  # Reduced for faster tests
        'd_layers': 2,  # Reduced for faster tests
        'd_ff': 48,  # 4 * d_model
        'dropout_ff': 0.1,
    }

@pytest.fixture
def sample_tensors(basic_config):
    """Generate sample tensors for testing."""
    config = basic_config
    return {
        'x': torch.rand(config['batch_size'], config['seq_len'], config['time_comp']),
        'y': torch.rand(config['batch_size'], config['seq_len'], 1),
        'p': torch.arange(0, config['seq_len']).view(1, config['seq_len']).repeat(config['batch_size'], 1),
        'val_time_emb': torch.rand(config['batch_size'], config['seq_len'], config['d_model']),
        'space_emb': torch.rand(config['batch_size'], config['seq_len'], config['d_model']),
    }

@pytest.fixture
def data_dir():
    """Return the data directory path if it exists."""
    data_path = PROJECT_ROOT / "data"
    return data_path if data_path.exists() else None

@pytest.fixture
def skip_if_no_data(data_dir):
    """Skip test if data directory doesn't exist."""
    if data_dir is None:
        pytest.skip("Data directory not found - skipping data-dependent test")
    return data_dir

@pytest.fixture
def mock_data_files(tmp_path):
    """Create mock data files for testing."""
    # Create mock numpy arrays
    X_np = np.random.rand(5, 8, 20)  # batch, features, sequence
    Y_np = np.random.rand(5, 8, 20)
    
    # Save to temporary directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    np.save(input_dir / "X_np.npy", X_np)
    np.save(input_dir / "Y_np.npy", Y_np)
    
    return {
        'input_dir': input_dir,
        'X_np': X_np,
        'Y_np': Y_np
    }
