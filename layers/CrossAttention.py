import torch
import torch.nn as nn

class MultiheadLatentAttention(nn.Module):
    """
    Multihead Latent Cross-Attention mechanism for token-level fusion.
    
    This module projects query and key/value inputs into a shared latent space
    where cross-attention is performed, enabling effective fusion of modalities
    with different dimensions.
    
    In the Chimera model:
    - Queries come from the iTransformer output
    - Keys and values come from the text features
    """
    def __init__(self, query_dim, key_dim, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"
        
        # Projection layers to latent space
        self.q_proj = nn.Linear(query_dim, latent_dim)
        self.k_proj = nn.Linear(key_dim, latent_dim)
        self.v_proj = nn.Linear(key_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, queries, keys, values, attention_mask=None):
        """
        Perform cross-attention in latent space
        
        Args:
            queries: Tensor from time series branch (B, L_q, query_dim)
            keys: Tensor from text branch (B, L_k, key_dim)
            values: Tensor from text branch (B, L_v, key_dim)
            Where L_q, L_k, and L_v are the sequence lengths for their respective tensors
            
        Returns:
            Fused features in latent space (B, L_q, latent_dim)
        """
        batch_size = queries.shape[0]
        
        # Project to latent space
        q = self.q_proj(queries)  # (batch_size, seq_len_q, latent_dim)
        k = self.k_proj(keys)     # (batch_size, seq_len_k, latent_dim)
        v = self.v_proj(values)   # (batch_size, seq_len_v, latent_dim)
        
        # Reshape for multihead attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_q, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_k, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_v, head_dim)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_q, seq_k)
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_q, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output # (batch, seq_q, d_latent) where seq_q is the variate dimension