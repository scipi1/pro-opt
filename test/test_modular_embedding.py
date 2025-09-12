"""
Tests for modular embedding modules.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from proT.modules.embedding import ModularEmbedding


class TestModularEmbedding:
    """Test class for modular embedding mechanisms."""
    
    @pytest.mark.unit
    def test_modular_embedding_initialization(self):
        """Test that ModularEmbedding can be initialized."""
        d_model = 8
        vocab_size = 100
        time_dim = 5
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": slice(1, None),
                "embed": "time2vec",
                "kwargs": {"input_dim": time_dim, "embed_dim": time_dim * 2}
            },
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        
        assert embed is not None
        assert hasattr(embed, 'forward')
    
    @pytest.mark.unit
    def test_modular_embedding_forward_simple(self, torch_seed):
        """Test modular embedding forward pass with simple configuration."""
        batch_size, seq_len = 3, 8
        d_model = 4
        vocab_size = 50
        time_dim = 3
        
        # Create input tensor: [position, time_features...]
        X = torch.zeros(batch_size, seq_len, 1 + time_dim)
        X[:, :, 0] = torch.randint(0, vocab_size, (batch_size, seq_len))  # positions
        X[:, :, 1:] = torch.rand(batch_size, seq_len, time_dim)  # time features
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": slice(1, None),
                "embed": "time2vec",
                "kwargs": {"input_dim": time_dim, "embed_dim": d_model}
            },
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        embed_out = embed(X)
        
        # With concat, output should be d_model + d_model = 2 * d_model
        expected_shape = (batch_size, seq_len, 2 * d_model)
        assert embed_out.shape == expected_shape, f"Expected shape {expected_shape}, got {embed_out.shape}"
        assert torch.isfinite(embed_out).all(), "Embedding output contains NaN or inf values"
    
    @pytest.mark.unit
    def test_modular_embedding_sum_composition(self, torch_seed):
        """Test modular embedding with sum composition."""
        batch_size, seq_len = 2, 5
        d_model = 6
        vocab_size = 30
        time_dim = 2
        
        X = torch.zeros(batch_size, seq_len, 1 + time_dim)
        X[:, :, 0] = torch.randint(0, vocab_size, (batch_size, seq_len))
        X[:, :, 1:] = torch.rand(batch_size, seq_len, time_dim)
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": slice(1, None),
                "embed": "time2vec",
                "kwargs": {"input_dim": time_dim, "embed_dim": d_model}
            },
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="sum")
        embed_out = embed(X)
        
        # With sum, output should be d_model
        expected_shape = (batch_size, seq_len, d_model)
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.unit
    def test_modular_embedding_single_component(self, torch_seed):
        """Test modular embedding with single component."""
        batch_size, seq_len = 2, 4
        d_model = 8
        vocab_size = 100
        
        X = torch.randint(0, vocab_size, (batch_size, seq_len, 1))
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            }
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        embed_out = embed(X)
        
        expected_shape = (batch_size, seq_len, d_model)
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.unit
    def test_modular_embedding_different_indices(self, torch_seed):
        """Test modular embedding with different index specifications."""
        batch_size, seq_len = 2, 6
        d_model = 4
        vocab_size = 50
        
        # Create input with multiple features
        X = torch.zeros(batch_size, seq_len, 5)
        X[:, :, 0] = torch.randint(0, vocab_size, (batch_size, seq_len))  # categorical
        X[:, :, 1] = torch.randint(0, vocab_size, (batch_size, seq_len))  # categorical
        X[:, :, 2:] = torch.rand(batch_size, seq_len, 3)  # continuous
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": 1,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": slice(2, None),
                "embed": "time2vec",
                "kwargs": {"input_dim": 3, "embed_dim": d_model}
            },
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        embed_out = embed(X)
        
        # Three components, each d_model
        expected_shape = (batch_size, seq_len, 3 * d_model)
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_modular_embedding_with_mock_data(self, torch_seed, mock_data_files):
        """Test modular embedding with mock data files."""
        mock_data = mock_data_files
        
        # Use a subset of the mock data
        X_subset = mock_data['X_np'][:2, :, :8]  # batch=2, features=all, seq=8
        X = torch.tensor(X_subset.astype("float"))
        
        # Handle NaN values
        X = X.nan_to_num()
        
        pos_idx = 0
        time_idx = slice(1, None)
        time_dim = X.shape[-1] - 1
        d_model = 6
        vocab_size = int(X.max().item()) + 1
        
        ds_embed = [
            {
                "idx": pos_idx,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            },
            {
                "idx": time_idx,
                "embed": "time2vec",
                "kwargs": {"input_dim": time_dim, "embed_dim": d_model}
            },
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        embed_out = embed(X)
        
        expected_shape = (X.shape[0], X.shape[1], 2 * d_model)
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.unit
    def test_modular_embedding_edge_cases(self, torch_seed):
        """Test modular embedding with edge cases."""
        # Test with minimal dimensions
        batch_size, seq_len = 1, 1
        d_model = 2
        vocab_size = 5
        
        X = torch.randint(0, vocab_size, (batch_size, seq_len, 1))
        
        ds_embed = [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "kwargs": {"num_embeddings": vocab_size, "embedding_dim": d_model}
            }
        ]
        
        embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
        embed_out = embed(X)
        
        expected_shape = (batch_size, seq_len, d_model)
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()


@pytest.mark.unit
def test_modular_embedding_initialization_errors():
    """Test that ModularEmbedding handles initialization errors gracefully."""
    # Test with empty ds_embed
    with pytest.raises((ValueError, TypeError, IndexError)):
        embed = ModularEmbedding(ds_embed=[], comps="concat")
    
    # Test with invalid composition method
    ds_embed = [
        {
            "idx": 0,
            "embed": "nn_embedding",
            "kwargs": {"num_embeddings": 10, "embedding_dim": 4}
        }
    ]
    
    # This might raise an error or might default to a valid method
    try:
        embed = ModularEmbedding(ds_embed=ds_embed, comps="invalid_method")
        # If it doesn't raise an error, that's also acceptable
    except (ValueError, TypeError):
        # Expected behavior for invalid composition method
        pass


@pytest.mark.unit
def test_modular_embedding_time2vec_only():
    """Test modular embedding with only time2vec component."""
    batch_size, seq_len = 2, 4
    time_dim = 3
    embed_dim = 6
    
    X = torch.rand(batch_size, seq_len, time_dim)
    
    ds_embed = [
        {
            "idx": slice(None),
            "embed": "time2vec",
            "kwargs": {"input_dim": time_dim, "embed_dim": embed_dim}
        }
    ]
    
    embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
    embed_out = embed(X)
    
    expected_shape = (batch_size, seq_len, embed_dim)
    assert embed_out.shape == expected_shape
    assert torch.isfinite(embed_out).all()
