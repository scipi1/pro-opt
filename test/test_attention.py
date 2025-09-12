"""
Tests for attention modules.
"""
import pytest
import torch
from proT.modules.attention import ScaledDotAttention, AttentionLayer
from proT.modules.extra_layers import UniformAttentionMask


class TestAttention:
    """Test class for attention mechanisms."""
    
    @pytest.mark.unit
    def test_attention_layer_initialization(self, basic_config):
        """Test that AttentionLayer can be initialized with valid parameters."""
        config = basic_config
        
        attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=config['d_model'],
            d_model_keys=config['d_model'],
            d_model_values=config['d_model'],
            d_queries_keys=config['d_queries_keys'],
            n_heads=config['n_heads'],
            mask_layer=UniformAttentionMask(),
            attention_dropout=0,
            dropout_qkv=config['dropout_qkv']
        )
        
        assert attention is not None
        assert attention.n_heads == config['n_heads']
        assert attention.d_queries_keys == config['d_queries_keys']
    
    @pytest.mark.unit
    def test_attention_forward_pass(self, torch_seed, basic_config):
        """Test attention forward pass with proper shapes."""
        config = basic_config
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        d_model = config['d_model']
        n_heads = config['n_heads']
        
        # Create input tensor
        x = torch.rand(batch_size, seq_len, d_model)
        
        # Initialize attention layer
        attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model,
            d_model_keys=d_model,
            d_model_values=d_model,
            d_queries_keys=config['d_queries_keys'],
            n_heads=n_heads,
            mask_layer=UniformAttentionMask(),
            attention_dropout=0,
            dropout_qkv=config['dropout_qkv']
        )
        
        # Forward pass
        out, score = attention(
            query=x,
            key=x,
            value=x,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=False
        )
        
        # Check output shapes
        expected_out_shape = (batch_size, seq_len, d_model)
        expected_score_shape = (batch_size, n_heads, seq_len, seq_len)
        
        assert out.shape == expected_out_shape, f"Expected output shape {expected_out_shape}, got {out.shape}"
        assert score.shape == expected_score_shape, f"Expected score shape {expected_score_shape}, got {score.shape}"
        
        # Check that outputs are valid tensors (no NaN or inf)
        assert torch.isfinite(out).all(), "Output contains NaN or inf values"
        assert torch.isfinite(score).all(), "Score contains NaN or inf values"
    
    @pytest.mark.unit
    def test_attention_single_head(self, torch_seed, basic_config):
        """Test attention with single head."""
        config = basic_config
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        d_model = config['d_model']
        n_heads = 1  # Single head
        
        x = torch.rand(batch_size, seq_len, d_model)
        
        attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model,
            d_model_keys=d_model,
            d_model_values=d_model,
            d_queries_keys=config['d_queries_keys'],
            n_heads=n_heads,
            mask_layer=UniformAttentionMask(),
            attention_dropout=0,
            dropout_qkv=config['dropout_qkv']
        )
        
        out, score = attention(
            query=x,
            key=x,
            value=x,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=False
        )
        
        # For single-head attention, score shape should be (B, 1, L, L)
        expected_out_shape = (batch_size, seq_len, d_model)
        expected_score_shape = (batch_size, 1, seq_len, seq_len)
        
        assert out.shape == expected_out_shape
        assert score.shape == expected_score_shape
    
    @pytest.mark.unit
    def test_attention_with_causal_mask(self, torch_seed, basic_config):
        """Test attention with causal masking."""
        config = basic_config
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        d_model = config['d_model']
        
        x = torch.rand(batch_size, seq_len, d_model)
        
        attention = AttentionLayer(
            attention=ScaledDotAttention,
            d_model_queries=d_model,
            d_model_keys=d_model,
            d_model_values=d_model,
            d_queries_keys=config['d_queries_keys'],
            n_heads=config['n_heads'],
            mask_layer=UniformAttentionMask(),
            attention_dropout=0,
            dropout_qkv=config['dropout_qkv']
        )
        
        out, score = attention(
            query=x,
            key=x,
            value=x,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=True
        )
        
        # Check that outputs have correct shapes
        assert out.shape == (batch_size, seq_len, d_model)
        assert score.shape == (batch_size, config['n_heads'], seq_len, seq_len)
        
        # Check that outputs are finite
        assert torch.isfinite(out).all()
        assert torch.isfinite(score).all()


@pytest.mark.unit
def test_scaled_dot_attention_direct():
    """Test ScaledDotAttention directly."""
    batch_size, seq_len, d_k = 2, 5, 8
    
    # Create Q, K, V tensors
    Q = torch.rand(batch_size, seq_len, d_k)
    K = torch.rand(batch_size, seq_len, d_k)
    V = torch.rand(batch_size, seq_len, d_k)
    
    # Create attention with required parameters
    from proT.modules.extra_layers import UniformAttentionMask
    attention = ScaledDotAttention(
        mask_layer=UniformAttentionMask(),
        attention_dropout=0.0,
        register_entropy=False,
        layer_name="test_attention"
    )
    
    # Forward pass
    result = attention(Q, K, V, mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False)
    
    # Handle different return formats
    if isinstance(result, tuple):
        if len(result) == 2:
            out, scores = result
        else:
            # If more than 2 values, take first two
            out, scores = result[0], result[1]
    else:
        # If single value, assume it's the output
        out = result
        scores = None
    
    # Check shapes
    assert out.shape == (batch_size, seq_len, d_k)
    
    if scores is not None:
        assert scores.shape == (batch_size, seq_len, seq_len)
        # Check that attention scores sum to 1 along the last dimension
        assert torch.allclose(scores.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
