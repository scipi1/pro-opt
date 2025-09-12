"""
Tests for encoder modules.
"""
import pytest
import torch
from proT.modules.encoder import Encoder, EncoderLayer
from proT.modules.attention import ScaledDotAttention, AttentionLayer
from proT.modules.extra_layers import Normalization, UniformAttentionMask


class TestEncoder:
    """Test class for encoder mechanisms."""
    
    @pytest.fixture
    def attention_layer(self, basic_config):
        """Create an attention layer for testing."""
        config = basic_config
        return AttentionLayer(
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
    
    @pytest.mark.unit
    def test_encoder_layer_initialization(self, basic_config, attention_layer):
        """Test that EncoderLayer can be initialized with valid parameters."""
        config = basic_config
        
        encoder_layer = EncoderLayer(
            global_attention=attention_layer,
            d_model_enc=config['d_model'],
            dropout_attn_out=config['dropout_attn_out'],
            activation=config['activation'],
            norm=config['norm']
        )
        
        assert encoder_layer is not None
        assert encoder_layer.d_model_enc == config['d_model']
    
    @pytest.mark.unit
    def test_encoder_initialization(self, basic_config, attention_layer):
        """Test that Encoder can be initialized with valid parameters."""
        config = basic_config
        
        encoder_layers = [
            EncoderLayer(
                global_attention=attention_layer,
                d_model_enc=config['d_model'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['e_layers'])
        ]
        
        encoder = Encoder(
            encoder_layers=encoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']) if config['use_final_norm'] else None,
            emb_dropout=config['dropout_emb']
        )
        
        assert encoder is not None
        assert len(encoder.encoder_layers) == config['e_layers']
    
    @pytest.mark.unit
    def test_encoder_forward_pass(self, torch_seed, basic_config, sample_tensors, attention_layer):
        """Test encoder forward pass with proper shapes."""
        config = basic_config
        tensors = sample_tensors
        
        encoder_layers = [
            EncoderLayer(
                global_attention=attention_layer,
                d_model_enc=config['d_model'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['e_layers'])
        ]
        
        encoder = Encoder(
            encoder_layers=encoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']) if config['use_final_norm'] else None,
            emb_dropout=config['dropout_emb']
        )
        
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        
        x, attn = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        # Check output shapes
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert x.shape == expected_shape, f"Expected output shape {expected_shape}, got {x.shape}"
        
        # Check attention outputs
        assert isinstance(attn, list), "Attention should be a list"
        assert len(attn) == config['e_layers'], f"Expected {config['e_layers']} attention tensors, got {len(attn)}"
        
        # Check that outputs are finite
        assert torch.isfinite(x).all(), "Encoder output contains NaN or inf values"
        for i, att in enumerate(attn):
            assert torch.isfinite(att).all(), f"Attention tensor {i} contains NaN or inf values"
    
    @pytest.mark.unit
    def test_encoder_without_final_norm(self, torch_seed, basic_config, sample_tensors, attention_layer):
        """Test encoder without final normalization."""
        config = basic_config
        tensors = sample_tensors
        
        encoder_layers = [
            EncoderLayer(
                global_attention=attention_layer,
                d_model_enc=config['d_model'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['e_layers'])
        ]
        
        encoder = Encoder(
            encoder_layers=encoder_layers,
            norm_layer=None,  # No final normalization
            emb_dropout=config['dropout_emb']
        )
        
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        
        x, attn = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert x.shape == expected_shape
        assert isinstance(attn, list)
        assert len(attn) == config['e_layers']
        assert torch.isfinite(x).all()
    
    @pytest.mark.unit
    def test_encoder_single_layer(self, torch_seed, basic_config, sample_tensors, attention_layer):
        """Test encoder with single layer."""
        config = basic_config
        tensors = sample_tensors
        
        encoder_layers = [
            EncoderLayer(
                global_attention=attention_layer,
                d_model_enc=config['d_model'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            )
        ]
        
        encoder = Encoder(
            encoder_layers=encoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']),
            emb_dropout=config['dropout_emb']
        )
        
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        
        x, attn = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert x.shape == expected_shape
        assert isinstance(attn, list)
        assert len(attn) == 1  # Single layer
        assert torch.isfinite(x).all()
    
    @pytest.mark.unit
    def test_encoder_different_activations(self, torch_seed, basic_config, sample_tensors, attention_layer):
        """Test encoder with different activation functions."""
        config = basic_config
        tensors = sample_tensors
        
        for activation in ['relu', 'gelu']:
            encoder_layers = [
                EncoderLayer(
                    global_attention=attention_layer,
                    d_model_enc=config['d_model'],
                    dropout_attn_out=config['dropout_attn_out'],
                    activation=activation,
                    norm=config['norm']
                )
            ]
            
            encoder = Encoder(
                encoder_layers=encoder_layers,
                norm_layer=Normalization(config['norm'], d_model=config['d_model']),
                emb_dropout=config['dropout_emb']
            )
            
            val_time_emb = tensors['val_time_emb']
            space_emb = tensors['space_emb']
            
            x, attn = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
            
            expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
            assert x.shape == expected_shape
            assert torch.isfinite(x).all()


@pytest.mark.unit
def test_encoder_layer_standalone():
    """Test EncoderLayer as standalone component."""
    batch_size, seq_len, d_model = 2, 5, 8
    
    # Create attention layer
    attention = AttentionLayer(
        attention=ScaledDotAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_model,
        n_heads=2,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0.1
    )
    
    encoder_layer = EncoderLayer(
        global_attention=attention,
        d_model_enc=d_model,
        dropout_attn_out=0.1,
        activation='relu',
        norm='layer'
    )
    
    # Create input
    val_time_emb = torch.rand(batch_size, seq_len, d_model)
    space_emb = torch.rand(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = encoder_layer(val_time_emb=val_time_emb, space_emb=space_emb)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(output).all()
    assert attention_weights is not None
