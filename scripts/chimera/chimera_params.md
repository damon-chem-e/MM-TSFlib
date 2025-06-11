# Chimera Hyperparameters

### Basic Config

- `--is_training`: Boolean flag (0 or 1) indicating whether to train the model (1) or run inference only (0)
- `--model_id`: Unique identifier for the model, used for saving checkpoints and results

### Multimodal Fusion Parameters
- `--gate_type`: Gate type to use ('mlp', 'linear', 'linear_norm', 'per_token_scalar', 'global_scalar')
- `--gate_hidden_dim`: If gate_type=='mlp', the hidden dimension of the gate. Optional, can be omitted.
- `--gate_regularization_lambda`: 
- `--llm_model`: LLM model to use ('LLAMA2', 'BERT', etc.)
- `--huggingface_token`: HuggingFace API token for model access

### Data Configuration

- `--data`: Name of the dataset to use
- `--root_path`: Root directory containing the data
- `--data_path`: Path to the data file relative to root_path
- `--text_path`: Path to text data file for multimodal fusion
- `--freq`: Frequency of the time series data ('h' for hourly, 't' for minutely, 's' for secondly)
- `--target`: Target variable to predict
- `--percent`: Percentage of data to use for training (e.g., 10, 20, 50, 100)

### Attention/Encoders

iTransformer

- `--num_layers`: Number of iTransformer encoder layers
- `--d_model`: Dim of model

Text self-attention

- `--num_layers_llm`: Number of text encoder layers. Can be 0.
- `--d_llm`: d_model for the text encoder layers.

Cross-attention

- `--d_latent`: Latent dimension for cross attention. Optional, can be omitted.
- `--fusion_heads`: Number cross-attention heads. Optional, can be omitted.

Post fusion attention
- `--post_fusion_layers`: Number of post fusion attention layers. Can be 0.

Final attention layers
- `--final_layers`: Number of final self attention layers. Can be 0.

Parameters shared among several attention blocks

- `--d_ff`: Dim of feedforward for encoers
- `--dropout`: Dropout for encoders/attentions
- `--n_heads`: Number of attention heads
- `--activation`: 'relu' or 'gelu'

### Sequence Configuration
- `--seq_len`: Length of time series input sequences
- `--pred_len`: Length of time series prediction horizon
- `--text_len`: Maximum length of text sequences

### Training Parameters
- `--train_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--patience`: Number of epochs to wait before early stopping
- `--learning_rate`: Learning rate for entire model training
- `--des`: Description of the experiment
- `--loss`: Loss function to use ('mse', 'mae', 'mape', 'smape', 'mase')
- `--lradj`: Learning rate adjustment strategy ('type1', 'type2')
- `--use_amp`: Whether to use automatic mixed precision (0 or 1)

### GPU Configuration
- `--use_gpu`: Whether to use GPU (0 or 1)
- `--gpu`: GPU device ID
- `--use_multi_gpu`: Whether to use multiple GPUs (0 or 1)
- `--devices`: Comma-separated list of GPU device IDs for multi-GPU training
