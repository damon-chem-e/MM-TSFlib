import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.CrossAttention import MultiheadLatentAttention
from layers.GatingMechanism import FeatureGate

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs # Store configs for helper methods
        self.gate_regularization = configs.gate_regularization_lambda > 0
        self.is_training = configs.is_training
        
        # Build model components 
        self._build_itransformer_components()
        self._build_text_self_attention()
        self._build_cross_attention()
        self._build_post_fusion_layers()
        self._build_gating_mechanism()
        self._build_final_layers()
        self._build_projection_head()
        
    def _build_itransformer_components(self):
        """Initializes the core iTransformer components (embedding and encoder)."""
        configs = self.configs
        self.enc_embedding = DataEmbedding_inverted(c_in=configs.seq_len, d_model=configs.d_model, 
                                                    freq=configs.freq, dropout=configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
    def _build_text_self_attention(self):
        """Inits the text processing components (optional self-attention following text embeddings)"""
        configs = self.configs
        self.num_layers_llm = configs.num_layers_llm
        self.text_encoder = None
        if self.num_layers_llm > 0:
            self.text_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        configs.d_llm, configs.n_heads),
                    configs.d_llm,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(self.num_layers_llm)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_llm)
        )
            
    def _build_cross_attention(self):
        """Initializes the cross-modal attention mechanism."""
        configs = self.configs
        self.d_latent = configs.d_latent if hasattr(configs, 'd_latent') else min(configs.d_model, configs.d_llm)
        self.cross_attention = MultiheadLatentAttention(
            query_dim=configs.d_model,
            key_dim=configs.d_llm,
            latent_dim=self.d_latent,
            num_heads = configs.fusion_heads if hasattr(configs, 'fusion_heads') else configs.n_heads,
            dropout=configs.dropout
        )

    def _build_post_fusion_layers(self):
        """Initializes the optional self-attention layers after cross-attention."""
        configs = self.configs
        self.post_fusion_layers = configs.post_fusion_layers
        self.post_fusion_encoder = None
        if self.post_fusion_layers > 0:
            self.post_fusion_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                        configs.d_latent, configs.n_heads),
                    configs.d_latent,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(self.post_fusion_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_latent)
        )
            
    def _build_gating_mechanism(self):
        """Initializes the gating mechanism based on the specific type."""
        configs = self.configs
        self.feature_gate = FeatureGate(
            fused_dim = self.d_latent,
            ts_dim=configs.d_model,
            gate_type=configs.gate_type,
            hidden_dim=configs.gate_hidden_dim if hasattr(configs, 'gate_hidden_dim') else 2*configs.d_model # Only used for mlp gate
        )
        
    def _build_final_layers(self):
        """Initializes the optional final self-attention layers after gating."""
        configs = self.configs
        self.final_layers = configs.final_layers
        self.final_encoder = None
        if self.final_layers > 0:
            self.final_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(mask_flag=False, attention_dropout=configs.dropout),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(self.final_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
            
    def _build_projection_head(self):
        """Initializes the projection head."""
        configs = self.configs
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    def forward_text_encoder(self, text_embeddings):
        """Process text embeddings through optional self-attention layers."""
        if self.text_encoder is not None:
            text_features, _ = self.text_encoder(text_embeddings, attn_mask=None)
        else:
            text_features = text_embeddings
        return text_features
    
    def forward_fusion_and_post_process(self, text_features, ts_features):
        """Perform cross-modal fusion and optional post-fusion self-attention.
        
        Args:
            text_features: Features from text encoder (B, L, d_llm)
            ts_features: Features from iTransformer (B, V, d_model)
            
        Returns:
            fused_latent_features: Fused features in latent space (B, L, latent_dim)
        """
        # Cross-attention fusion
        fused_latent_features = self.cross_attention(
            queries=ts_features,    # Time series as query
            keys=text_features,     # Text as key
            values=text_features    # Text as value
        )
        
        # Post-fusion self-attention if specified
        if self.post_fusion_encoder is not None:
            fused_latent_features, _ = self.post_fusion_encoder(fused_latent_features, attn_mask=None)
            
        return fused_latent_features
    
    def forward_gating_fusion(self, fused_latent_features, ts_features):
        """Apply gating mechanism to control fusion strength.
        
        Args:
            fused_latent_features: Features from cross-attention/post-fusion (B, L, latent_dim)
            ts_features: Original time series features from iTransformer (B, L, d_model)
            
        Returns:
            gated_output: Combined features in time series dimension (B, L, d_model)
            gate_value: The computed gate (G or alpha) for potential regularization (B, L, d_model)
        """
        gated_output, gate_value = self.feature_gate(
            fused_latent_features, ts_features
        )
        return gated_output, gate_value
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings=None):
        gate_value = None # Initialize gate value
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Time series embedding and encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        ts_features, _ = self.encoder(enc_out, attn_mask=None)
        
        # Process text if available for multimodal fusion
        if text_embeddings is not None:
            # Text processing
            text_features = self.forward_text_encoder(text_embeddings)
            
            # Cross-modal fusion and optional post-processing
            fused_latent_features = self.forward_fusion_and_post_process(text_features, ts_features)
            
            # Gated fusion with time series features (includes dimension projection)
            final_features, gate_value = self.forward_gating_fusion(fused_latent_features, ts_features)
            
            # Final transformer layers if specified
            if self.final_encoder is not None:
                final_features, _ = self.final_encoder(final_features, attn_mask=None)
        else:
            final_features = ts_features
        
        # Task-specific prediction
        dec_out = self.projection(final_features).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out, gate_value
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings=None, mask=None):
        # Long term forecasting task
        # if self.gate_regularization:
        #     dec_out, gate_value = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
        #     return dec
        # else:
        #     dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
        
        # return dec_out, gate_value # Return gate for regularization loss
    
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
    