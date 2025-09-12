"""
Tests for embedding modules.
"""
import pytest
import torch
from proT.modules.simple_embedding import Embedding


class TestEmbedding:
    """Test class for embedding mechanisms."""
    
    @pytest.mark.unit
    def test_embedding_initialization(self, basic_config):
        """Test that Embedding can be initialized with valid parameters."""
        config = basic_config
        
        embed = Embedding(
            d_value=1,
            d_time=config['time_comp'],
            d_model=config['d_model'],
            use_val=True
        )
        
        assert embed is not None
        assert embed.d_model == config['d_model']
        assert embed.d_time == config['time_comp']
    
    @pytest.mark.unit
    def test_embedding_forward_pass(self, torch_seed, basic_config, sample_tensors):
        """Test embedding forward pass with proper shapes."""
        config = basic_config
        tensors = sample_tensors
        
        embed = Embedding(
            d_value=1,
            d_time=config['time_comp'],
            d_model=config['d_model'],
            use_val=True
        )
        
        # Use sample tensors from fixture
        x = tensors['x']  # (batch_size, seq_len, time_comp)
        y = tensors['y']  # (batch_size, seq_len, 1)
        p = tensors['p']  # (batch_size, seq_len)
        
        # Add dimension to p for embedding
        p = p.unsqueeze(-1)
        
        embed_out = embed(x=x, y=y, p=p)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert embed_out.shape == expected_shape, f"Expected shape {expected_shape}, got {embed_out.shape}"
        
        # Check that output is finite
        assert torch.isfinite(embed_out).all(), "Embedding output contains NaN or inf values"
    
    @pytest.mark.unit
    def test_embedding_without_value(self, torch_seed, basic_config, sample_tensors):
        """Test embedding without value component."""
        config = basic_config
        tensors = sample_tensors
        
        embed = Embedding(
            d_value=1,
            d_time=config['time_comp'],
            d_model=config['d_model'],
            use_val=False  # Don't use value
        )
        
        x = tensors['x']
        y = tensors['y']
        p = tensors['p'].unsqueeze(-1)
        
        embed_out = embed(x=x, y=y, p=p)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.unit
    def test_embedding_with_full_config(self, torch_seed, basic_config, sample_tensors):
        """Test embedding with full configuration parameters."""
        config = basic_config
        tensors = sample_tensors
        
        embed = Embedding(
            d_value=1,
            d_time=config['time_comp'],
            d_model=config['d_model'],
            time_emb_dim=config['time_comp'],
            d_var_emb=10,
            var_vocab_siz=1000,
            is_encoder=True,
            embed_method="spatio-temporal",
            dropout_data=None,
            max_seq_len=1600,
            use_given=True,
            use_val=True
        )
        
        x = tensors['x']
        y = tensors['y']
        p = tensors['p'].unsqueeze(-1)
        
        embed_out = embed(x=x, y=y, p=p)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert embed_out.shape == expected_shape
        assert torch.isfinite(embed_out).all()
    
    @pytest.mark.unit
    def test_embedding_different_batch_sizes(self, torch_seed, basic_config):
        """Test embedding with different batch sizes."""
        config = basic_config
        
        embed = Embedding(
            d_value=1,
            d_time=config['time_comp'],
            d_model=config['d_model'],
            use_val=True
        )
        
        # Test with different batch sizes
        for batch_size in [1, 3, 5]:
            x = torch.rand(batch_size, config['seq_len'], config['time_comp'])
            y = torch.rand(batch_size, config['seq_len'], 1)
            p = torch.arange(0, config['seq_len']).view(1, config['seq_len']).repeat(batch_size, 1).unsqueeze(-1)
            
            embed_out = embed(x=x, y=y, p=p)
            
            expected_shape = (batch_size, config['seq_len'], config['d_model'])
            assert embed_out.shape == expected_shape
            assert torch.isfinite(embed_out).all()


@pytest.mark.unit
def test_embedding_minimal_config():
    """Test embedding with minimal configuration."""
    batch_size, seq_len, time_comp, d_model = 2, 5, 3, 8
    
    embed = Embedding(
        d_value=1,
        d_time=time_comp,
        d_model=d_model,
        use_val=True
    )
    
    x = torch.rand(batch_size, seq_len, time_comp)
    y = torch.rand(batch_size, seq_len, 1)
    p = torch.arange(0, seq_len).view(1, seq_len).repeat(batch_size, 1).unsqueeze(-1)
    
    embed_out = embed(x=x, y=y, p=p)
    
    assert embed_out.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(embed_out).all()


@pytest.mark.unit
def test_embedding_edge_cases():
    """Test embedding with edge cases."""
    # Test with sequence length of 1
    batch_size, seq_len, time_comp, d_model = 1, 1, 2, 4
    
    embed = Embedding(
        d_value=1,
        d_time=time_comp,
        d_model=d_model,
        use_val=True
    )
    
    x = torch.rand(batch_size, seq_len, time_comp)
    y = torch.rand(batch_size, seq_len, 1)
    p = torch.zeros(batch_size, seq_len, 1)
    
    embed_out = embed(x=x, y=y, p=p)
    
    assert embed_out.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(embed_out).all()
