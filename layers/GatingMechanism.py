import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGate(nn.Module):
    """
    Gating mechanism to control the fusion of multimodal features with multiple options.
    
    Projects fused features from latent dimension to time series dimension
    and computes a gate value to combine with original time series features.
    
    Options for gate computation (`gate_type`):
    - 'mlp': Original two-layer MLP gate network. Computes gate $$ \alpha \in [0, 1]^{B \times V \times d_t} $$.
            $$ 
            \alpha = \sigma(W_{g2} \cdot \text{ReLU}(W_{g1} C + b_{g1}) + b_{g2}) 
            $$
            
    - 'linear': Simple adaptive gate (alpha) using one linear layer on concatenated features. 
            Computes gate $$ \alpha \in [0, 1]^{B \times V \times d_t} $$.
            $$
            \alpha = \sigma(W_g C + b_g)
            $$
            
    - 'linear_norm': Lightweight linear gate using LayerNorm.
            Computes gate $$ \alpha \in [0, 1]^{B \times V \times d_t} $$.
            $$
            \alpha = \sigma(\text{LayerNorm}(W_g C + b_g))
            $$
            
    - 'per_token_scalar': Linear gate producing one scalar per token.
            Computes gate $$ \alpha \in [0, 1]^{B \times V \times 1} $$.
            $$
            \alpha = \sigma (W_g C + b_g)
            $$
            
    - 'global_scalar': Mean-pooled inputs to a linear gate producing one scalar per batch.
            Computes gate $$ \alpha \in [0, 1]^{B \times 1 \times 1} $$.
            $$
            \alpha = \sigma (W_g \bar C + b_g)
            $$
            
    """
    def __init__(self, fused_dim, ts_dim, gate_type='per_token_scalar', hidden_dim=None):
        """
        Inputs:
        - **Fused Latent Features**: $$ F \in \mathbb{R}^{B \times V \times d_f} $$, the output of the cross-attention mechanism (potentially after post-fusion self-attention), where $$ d_f $$ is `latent_dim`.
        - **Time Series Features**: $$ T \in \mathbb{R}^{B \times V \times d_t} $$, the output of the iTransformer encoder, where $$ d_t $$ is `d_model`.
        - **Hidden Dimension**: $$ d_h $$, typically `gate_hidden_dim` (defaulting to $$ 2 \times d_t $$), used only for the `mlp` gate.
        - **Batch size**: $$ B $$
        - **Variate dimension**: $$ V $$
        
        The fused features $$ F $$ are projected to $$ F' \in \mathbb{R}^{B \times V \times d_t} $$:
        $$
        F' = \text{FusedProjection}(F) = W_{p2} \cdot \text{ReLU}(W_{p1} F + b_{p1}) + b_{p2}
        $$
        
        """
        super(FeatureGate, self).__init__()
        self.gate_type = gate_type
        self.fused_dim = fused_dim # latent_dim from previous step
        self.ts_dim = ts_dim # d_model

        # --- Projection Layer (Common to all gate types) ---
        # Uses hidden_dim only if gate_type is 'mlp' (for compatibility)
        proj_hidden_dim = hidden_dim if hidden_dim is not None else 2 * ts_dim
        self.fused_projection = nn.Sequential(
            nn.Linear(fused_dim, proj_hidden_dim), # Use proj_hidden_dim for potential MLP compatibility
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, ts_dim)
        )

        # --- Gate Computation Network (Varies by gate_type) ---
        if gate_type == 'mlp':
            if hidden_dim is None:
                hidden_dim = 2 * ts_dim
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'linear':
            # Computes alpha = sigma(Linear([F, T]))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'linear_norm':
            # Computes gate = sigma(LayerNorm(Linear([F, T])))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, ts_dim),
                nn.LayerNorm(ts_dim), # Apply LayerNorm before sigmoid
                nn.Sigmoid()
            )
        elif gate_type == 'per_token_scalar' or gate_type == 'global_scalar': # Use same computation
            # Computes beta = sigma(Linear([F+T, 1]))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}. Choose 'mlp', 'linear', 'linear_norm', 'per_token_scalar', 'global_scalar'.")

    def forward(self, fused_latent_features, ts_features):
        """
        Apply gating mechanism to control information flow.
        
        Args:
            fused_latent_features: Features from cross-attention/post-fusion (B, L, fused_dim)
            ts_features: Original time series features from iTransformer (B, L, ts_dim)
            
        Returns:
            gated_output: Combined features with learned gating (B, L, ts_dim)
            gate_value: The computed gate for potential regularization. 
                        For per_token_scalar, returns (B, L, 1)
                        For global_scalar, returns (B, 1, 1)
                        For all other gates, returns (B, L, ts_dim)
        """
        # Project fused features from latent dimension to time series dimension
        projected_fused = self.fused_projection(fused_latent_features) # F' with shape (B, L, ts_dim)
        
        # Compute gate value (alpha)
        # Input features for gate computation need the pre-projection fused features (F)
        concat_features = torch.cat([fused_latent_features, ts_features], dim=-1) # (F, T)

        if self.gate_type == 'global_scalar':
            # Compute mean pool
            # $$\bar C = \text{Mean}_L [C]$$
            pooled = concat_features.mean(dim=1)
            beta = self.gate_network(pooled)
            gate_value = beta.unsqueeze(1)
        else:
            # Gate value (alpha)
            # If per_token_scalar, in shape (B, L, 1)
            # If global_scalar, in shape (B, 1, 1)
            # Otherwise, in shape (B, L, d_t)
            gate_value = self.gate_network(concat_features)
        
        # Apply the gate to the time series features and fused projected features
        # Final gated output $$ O \in \mathbb{R}^{B \times V \times d_t} $$ same for all gates
        
        # $$ O = \alpha \odot T + (1 - \alpha) \odot F' $$
        gated_output = gate_value * ts_features + (1 - gate_value) * projected_fused
        
        return gated_output, gate_value