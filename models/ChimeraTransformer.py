import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.CrossAttention import MultiheadLatentAttention
from layers.GatingMechanism import FeatureGate
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.configs = configs  # Store configs for helper methods
        self.gate_regularization = configs.gate_regularization_lambda > 0

        # Build model components modularly
        self._build_itransformer_components()
        self._build_text_fusion_components()
        self._build_cross_attention()
        self._build_post_fusion_layers()
        self._build_gating_mechanism()
        self._build_final_layers()
        self._build_projection_head()

    def _build_itransformer_components(self):
        """Initializes the core iTransformer components (embedding and encoder)."""
        configs = self.configs
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=self.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def _build_text_fusion_components(self):
        """Initializes the text processing components (optional self-attention)."""
        configs = self.configs
        self.text_fusion_layers = configs.text_fusion_layers
        self.text_encoder = None
        if self.text_fusion_layers > 0:
            self.text_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=self.output_attention), configs.d_llm, configs.n_heads),
                        configs.d_llm,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(self.text_fusion_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_llm)
            )

    def _build_cross_attention(self):
        """Initializes the cross-modal attention mechanism."""
        configs = self.configs
        self.latent_dim = configs.latent_dim if hasattr(configs, 'latent_dim') else min(configs.d_model, configs.d_llm)
        self.cross_attention = MultiheadLatentAttention(
            query_dim=configs.d_llm,       # Text features dimension
            key_dim=configs.d_model,       # Time series features dimension
            latent_dim=self.latent_dim,
            num_heads=configs.fusion_heads if hasattr(configs, 'fusion_heads') else configs.n_heads,
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
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=self.output_attention), self.latent_dim, configs.n_heads),
                        self.latent_dim,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(self.post_fusion_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.latent_dim)
            )

    def _build_gating_mechanism(self):
        """Initializes the gating mechanism based on the specified type."""
        configs = self.configs
        self.feature_gate = FeatureGate(
            fused_dim=self.latent_dim,     # Dimension of latent space from cross-attention/post-fusion
            ts_dim=configs.d_model,        # Dimension of original time series features
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
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=self.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for _ in range(self.final_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

    def _build_projection_head(self):
        """Initializes the task-specific projection head."""
        configs = self.configs
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        else:
            # Default or raise error for unsupported task
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
            ts_features: Features from iTransformer (B, L, d_model)
            
        Returns:
            fused_latent_features: Fused features in latent space (B, L, latent_dim)
        """
        # Cross-attention fusion
        fused_latent_features = self.cross_attention(
            queries=text_features,  # Text as query
            keys=ts_features,       # Time series as key
            values=ts_features      # Time series as value
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
        
        if self.gate_regularization and self.training:
            return dec_out, gate_value
        else:
            return dec_out

    # Implement other task methods (imputation, anomaly_detection, classification) similarly
    # by adapting the existing iTransformer implementations with the fusion components

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings=None, mask=None):
        # Handle different tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.gate_regularization and self.training:
                dec_out, gate_value = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
                return dec_out, gate_value # Return gate for regularization loss
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
                return dec_out  # [B, L, D]
        
        elif self.task_name == 'imputation':
            # Adapt imputation method similarly, potentially returning gate value
            pass 
        elif self.task_name == 'anomaly_detection':
            # Adapt anomaly detection method similarly
            pass
        elif self.task_name == 'classification':
            # Adapt classification method similarly
            pass
        
        # Fallback or error for unsupported task
        return None 