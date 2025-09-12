"""
Simple embedding class for testing purposes.
This provides a basic embedding interface compatible with the original tests.
"""
import torch
import torch.nn as nn
from .embedding_layers import Time2Vec, SinusoidalPosition, identity_emb, nn_embedding, linear_emb


class Embedding(nn.Module):
    """
    Simple embedding class for testing purposes.
    This is a simplified version that combines value, time, and position embeddings.
    """
    
    def __init__(
        self,
        d_value: int = 1,
        d_time: int = 6,
        d_model: int = 100,
        time_emb_dim: int = 6,
        d_var_emb: int = 10,
        var_vocab_siz: int = 1000,
        is_encoder: bool = True,
        embed_method: str = "spatio-temporal",
        dropout_data: float = None,
        max_seq_len: int = 1600,
        use_given: bool = True,
        use_val: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_time = d_time
        self.d_value = d_value
        self.use_val = use_val
        self.embed_method = embed_method
        
        # Time embedding
        if d_time > 0:
            self.time_embedding = Time2Vec(
                input_dim=d_time,
                embed_dim=time_emb_dim,
                device=device
            )
            time_embed_dim = time_emb_dim
        else:
            self.time_embedding = None
            time_embed_dim = 0
        
        # Value embedding (identity)
        if use_val and d_value > 0:
            self.value_embedding = identity_emb(device=device)
            value_embed_dim = d_value
        else:
            self.value_embedding = None
            value_embed_dim = 0
        
        # Position embedding
        self.position_embedding = SinusoidalPosition(
            max_pos=max_seq_len,
            embed_dim=d_model,
            device=device
        )
        
        # Combine embeddings to d_model
        total_embed_dim = time_embed_dim + value_embed_dim
        if total_embed_dim > 0:
            self.projection = nn.Linear(total_embed_dim, d_model)
        else:
            self.projection = None
        
        # Dropout
        if dropout_data is not None:
            self.dropout = nn.Dropout(dropout_data)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor, y: torch.Tensor = None, p: torch.Tensor = None):
        """
        Forward pass for embedding.
        
        Args:
            x: Time features tensor (batch_size, seq_len, d_time)
            y: Value tensor (batch_size, seq_len, d_value) 
            p: Position tensor (batch_size, seq_len, 1)
        
        Returns:
            Embedded tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.shape[:2]
        embeddings = []
        
        # Time embedding
        if self.time_embedding is not None and x is not None:
            time_emb = self.time_embedding(x)
            embeddings.append(time_emb)
        
        # Value embedding
        if self.value_embedding is not None and y is not None and self.use_val:
            value_emb = self.value_embedding(y)
            embeddings.append(value_emb)
        
        # Combine embeddings
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            if self.projection is not None:
                combined = self.projection(combined)
        else:
            combined = torch.zeros(batch_size, seq_len, self.d_model)
        
        # Add position embedding
        if p is not None:
            # Convert position to proper format if needed
            if p.dim() == 3 and p.shape[-1] == 1:
                p = p.squeeze(-1)
            pos_emb = self.position_embedding(p)
            combined = combined + pos_emb
        
        # Apply dropout
        if self.dropout is not None:
            combined = self.dropout(combined)
        
        return combined
