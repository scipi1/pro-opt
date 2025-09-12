"""
Tests for decoder modules.
"""
import pytest
import torch
from proT.modules.decoder import Decoder, DecoderLayer
from proT.modules.encoder import Encoder, EncoderLayer
from proT.modules.attention import ScaledDotAttention, AttentionLayer
from proT.modules.extra_layers import Normalization, UniformAttentionMask


class TestDecoder:
    """Test class for decoder mechanisms."""
    
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
    
    @pytest.fixture
    def encoder(self, basic_config, attention_layer):
        """Create an encoder for testing decoder cross-attention."""
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
        
        return Encoder(
            encoder_layers=encoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']) if config['use_final_norm'] else None,
            emb_dropout=config['dropout_emb']
        )
    
    @pytest.mark.unit
    def test_decoder_layer_initialization(self, basic_config, attention_layer):
        """Test that DecoderLayer can be initialized with valid parameters."""
        config = basic_config
        
        decoder_layer = DecoderLayer(
            global_self_attention=attention_layer,
            local_self_attention=None,
            global_cross_attention=attention_layer,
            local_cross_attention=None,
            d_model_dec=config['d_model'],
            time_windows=None,
            time_window_offset=None,
            d_ff=config['d_ff'],
            d_yt=None,
            d_yc=None,
            dropout_ff=config['dropout_ff'],
            dropout_attn_out=config['dropout_attn_out'],
            activation=config['activation'],
            norm=config['norm']
        )
        
        assert decoder_layer is not None
        assert decoder_layer.d_model_dec == config['d_model']
    
    @pytest.mark.unit
    def test_decoder_initialization(self, basic_config, attention_layer):
        """Test that Decoder can be initialized with valid parameters."""
        config = basic_config
        
        decoder_layers = [
            DecoderLayer(
                global_self_attention=attention_layer,
                local_self_attention=None,
                global_cross_attention=attention_layer,
                local_cross_attention=None,
                d_model_dec=config['d_model'],
                time_windows=None,
                time_window_offset=None,
                d_ff=config['d_ff'],
                d_yt=None,
                d_yc=None,
                dropout_ff=config['dropout_ff'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['d_layers'])
        ]
        
        decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']) if config['use_final_norm'] else None,
            emb_dropout=config['dropout_emb']
        )
        
        assert decoder is not None
        assert len(decoder.decoder_layers) == config['d_layers']
    
    @pytest.mark.unit
    def test_decoder_forward_pass(self, torch_seed, basic_config, sample_tensors, attention_layer, encoder):
        """Test decoder forward pass with proper shapes."""
        config = basic_config
        tensors = sample_tensors
        
        # Get encoder output for cross-attention
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        encoder_output, _ = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        # Create decoder
        decoder_layers = [
            DecoderLayer(
                global_self_attention=attention_layer,
                local_self_attention=None,
                global_cross_attention=attention_layer,
                local_cross_attention=None,
                d_model_dec=config['d_model'],
                time_windows=None,
                time_window_offset=None,
                d_ff=config['d_ff'],
                d_yt=None,
                d_yc=None,
                dropout_ff=config['dropout_ff'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['d_layers'])
        ]
        
        decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']) if config['use_final_norm'] else None,
            emb_dropout=config['dropout_emb']
        )
        
        # Forward pass
        y, attns = decoder(val_time_emb=val_time_emb, space_emb=space_emb, cross=encoder_output)
        
        # Check output shapes
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert y.shape == expected_shape, f"Expected output shape {expected_shape}, got {y.shape}"
        
        # Check attention outputs
        assert isinstance(attns, list), "Attention should be a list"
        assert len(attns) == config['d_layers'], f"Expected {config['d_layers']} attention tensors, got {len(attns)}"
        
        # Check that outputs are finite
        assert torch.isfinite(y).all(), "Decoder output contains NaN or inf values"
        for i, att in enumerate(attns):
            assert torch.isfinite(att).all(), f"Attention tensor {i} contains NaN or inf values"
    
    @pytest.mark.unit
    def test_decoder_without_cross_attention(self, torch_seed, basic_config, sample_tensors, attention_layer):
        """Test decoder without cross-attention (self-attention only)."""
        config = basic_config
        tensors = sample_tensors
        
        decoder_layers = [
            DecoderLayer(
                global_self_attention=attention_layer,
                local_self_attention=None,
                global_cross_attention=None,  # No cross-attention
                local_cross_attention=None,
                d_model_dec=config['d_model'],
                time_windows=None,
                time_window_offset=None,
                d_ff=config['d_ff'],
                d_yt=None,
                d_yc=None,
                dropout_ff=config['dropout_ff'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['d_layers'])
        ]
        
        decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']),
            emb_dropout=config['dropout_emb']
        )
        
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        
        y, attns = decoder(val_time_emb=val_time_emb, space_emb=space_emb, cross=None)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert y.shape == expected_shape
        assert isinstance(attns, list)
        assert len(attns) == config['d_layers']
        assert torch.isfinite(y).all()
    
    @pytest.mark.unit
    def test_decoder_single_layer(self, torch_seed, basic_config, sample_tensors, attention_layer, encoder):
        """Test decoder with single layer."""
        config = basic_config
        tensors = sample_tensors
        
        # Get encoder output
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        encoder_output, _ = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        decoder_layers = [
            DecoderLayer(
                global_self_attention=attention_layer,
                local_self_attention=None,
                global_cross_attention=attention_layer,
                local_cross_attention=None,
                d_model_dec=config['d_model'],
                time_windows=None,
                time_window_offset=None,
                d_ff=config['d_ff'],
                d_yt=None,
                d_yc=None,
                dropout_ff=config['dropout_ff'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            )
        ]
        
        decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=Normalization(config['norm'], d_model=config['d_model']),
            emb_dropout=config['dropout_emb']
        )
        
        y, attns = decoder(val_time_emb=val_time_emb, space_emb=space_emb, cross=encoder_output)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert y.shape == expected_shape
        assert isinstance(attns, list)
        assert len(attns) == 1  # Single layer
        assert torch.isfinite(y).all()
    
    @pytest.mark.unit
    def test_decoder_without_final_norm(self, torch_seed, basic_config, sample_tensors, attention_layer, encoder):
        """Test decoder without final normalization."""
        config = basic_config
        tensors = sample_tensors
        
        # Get encoder output
        val_time_emb = tensors['val_time_emb']
        space_emb = tensors['space_emb']
        encoder_output, _ = encoder(val_time_emb=val_time_emb, space_emb=space_emb)
        
        decoder_layers = [
            DecoderLayer(
                global_self_attention=attention_layer,
                local_self_attention=None,
                global_cross_attention=attention_layer,
                local_cross_attention=None,
                d_model_dec=config['d_model'],
                time_windows=None,
                time_window_offset=None,
                d_ff=config['d_ff'],
                d_yt=None,
                d_yc=None,
                dropout_ff=config['dropout_ff'],
                dropout_attn_out=config['dropout_attn_out'],
                activation=config['activation'],
                norm=config['norm']
            ) for _ in range(config['d_layers'])
        ]
        
        decoder = Decoder(
            decoder_layers=decoder_layers,
            norm_layer=None,  # No final normalization
            emb_dropout=config['dropout_emb']
        )
        
        y, attns = decoder(val_time_emb=val_time_emb, space_emb=space_emb, cross=encoder_output)
        
        expected_shape = (config['batch_size'], config['seq_len'], config['d_model'])
        assert y.shape == expected_shape
        assert isinstance(attns, list)
        assert len(attns) == config['d_layers']
        assert torch.isfinite(y).all()


@pytest.mark.unit
def test_decoder_layer_standalone():
    """Test DecoderLayer as standalone component."""
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
    
    decoder_layer = DecoderLayer(
        global_self_attention=attention,
        local_self_attention=None,
        global_cross_attention=attention,
        local_cross_attention=None,
        d_model_dec=d_model,
        time_windows=None,
        time_window_offset=None,
        d_ff=32,  # 4 * d_model
        d_yt=None,
        d_yc=None,
        dropout_ff=0.1,
        dropout_attn_out=0.1,
        activation='relu',
        norm='layer'
    )
    
    # Create inputs
    val_time_emb = torch.rand(batch_size, seq_len, d_model)
    space_emb = torch.rand(batch_size, seq_len, d_model)
    cross = torch.rand(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = decoder_layer(
        val_time_emb=val_time_emb, 
        space_emb=space_emb, 
        cross=cross
    )
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(output).all()
    assert attention_weights is not None
