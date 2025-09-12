# ProT Test Suite

This directory contains a comprehensive test suite for the ProT (Probabilistic Transformer) project, refactored to work professionally with pytest.

## Overview

The test suite has been completely refactored from the original ad-hoc test scripts to a professional pytest-based testing framework with proper:

- **Test structure**: Organized test classes and functions
- **Fixtures**: Shared setup and configuration
- **Assertions**: Proper test assertions instead of print statements
- **Error handling**: Graceful handling of missing dependencies
- **Independence**: Tests can run independently without coupling
- **Markers**: Unit, integration, and slow test markers

## Test Files

### Core Module Tests
- `test_attention.py` - Tests for attention mechanisms (ScaledDotAttention, AttentionLayer)
- `test_embedding.py` - Tests for embedding modules (simple embedding for testing)
- `test_encoder.py` - Tests for encoder components (Encoder, EncoderLayer)
- `test_decoder.py` - Tests for decoder components (Decoder, DecoderLayer)
- `test_modular_embedding.py` - Tests for modular embedding system

### Integration Tests
- `test_model.py` - Tests for the main ProT model
- `test_predicted_mask.py` - Tests for prediction with masking functionality

### Configuration
- `conftest.py` - Shared fixtures and pytest configuration
- `pytest.ini` - Pytest settings and markers

## Running Tests

### Install Dependencies
```bash
pip install pytest pytest-mock
```

### Run All Tests
```bash
pytest test/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest test/ -m unit

# Integration tests only  
pytest test/ -m integration

# Exclude slow tests
pytest test/ -m "not slow"
```

### Run Specific Test Files
```bash
pytest test/test_attention.py -v
pytest test/test_embedding.py -v
```

### Run Specific Tests
```bash
pytest test/test_attention.py::test_scaled_dot_attention_direct -v
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring multiple components
- `@pytest.mark.slow` - Slow-running tests

## Fixtures

### Available Fixtures (from conftest.py)
- `torch_seed` - Sets reproducible random seeds
- `basic_config` - Basic configuration parameters
- `sample_tensors` - Pre-generated sample tensors
- `mock_data_files` - Mock data files for testing
- `data_dir` - Real data directory (if available)
- `skip_if_no_data` - Skip tests if data not available

## Key Improvements from Original Tests

### Before (Original)
- Manual `sys.path.append()` for imports
- `main()` functions with print statements
- No proper assertions
- Coupled tests importing from each other
- Hard-coded paths to external data
- No error handling for missing dependencies

### After (Professional)
- Clean imports using proper Python path
- Proper test classes and functions
- Comprehensive assertions with meaningful error messages
- Independent tests with shared fixtures
- Graceful handling of missing external dependencies
- Professional pytest structure with markers and configuration

## External Dependencies

Tests handle external dependencies gracefully:
- Missing data files are handled with `@pytest.mark.skipif` or mock data
- Missing modules are caught and tests are skipped appropriately
- Integration tests that require full model setup are marked and can be skipped

## Test Coverage

The test suite covers:
- Attention mechanisms
- Embedding layers
- Encoder components
- Decoder components
- Model initialization and forward passes
- Modular embedding system
- Masking functionality
- Edge cases and error conditions

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py` when possible
4. Add proper assertions with descriptive messages
5. Handle external dependencies gracefully
6. Keep tests independent and focused

## Example Test Structure

```python
import pytest
import torch
from proT.modules.your_module import YourClass

class TestYourClass:
    """Test class for YourClass."""
    
    @pytest.mark.unit
    def test_initialization(self, basic_config):
        """Test that YourClass can be initialized."""
        obj = YourClass(**basic_config)
        assert obj is not None
    
    @pytest.mark.unit
    def test_forward_pass(self, torch_seed, sample_tensors):
        """Test forward pass with proper shapes."""
        obj = YourClass()
        result = obj(sample_tensors['x'])
        assert result.shape == expected_shape
        assert torch.isfinite(result).all()
```

This professional test suite ensures code quality, catches regressions, and provides confidence in the ProT model implementation.
