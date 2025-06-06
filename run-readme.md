# MM-TSFlib Run Options Documentation

This document provides a comprehensive guide to all options available in `run.py` and how they are used within the MM-TSFlib library.

## Basic Configuration

### Task Configuration
- `--task_name`: Specifies the type of task to perform. Options include:
  - `long_term_forecast`: Long-term time series forecasting
    - Forecasts using arbitrary `pred_len` passed as parameter. Intended to have long `seq_len`. Multi-modal. Is the entire MM-TSF model (from frozen LLM to everything after).
  - `short_term_forecast`: Short-term time series forecasting
    - Matches M4 competition. Uses M4 competition forecast horizon, fixed so context = 2x horizon, uni-modal (just benchmarking existing TS models), outputs loss and predictions in M4-style csv.
  - `imputation`: Time series imputation
  - `anomaly_detection`: Anomaly detection in time series
  - `classification`: Time series classification

- `--is_training`: Boolean flag (0 or 1) indicating whether to train the model (1) or run inference only (0)
- `--model_id`: Unique identifier for the model, used for saving checkpoints and results
- `--model`: Name of the model architecture to use (e.g., 'iTransformer', 'PatchTST')

### Data Configuration
- `--data`: Name of the dataset to use
- `--root_path`: Root directory containing the data
- `--data_path`: Path to the data file relative to root_path
- `--text_path`: Path to text data file for multimodal fusion
- `--freq`: Frequency of the time series data ('h' for hourly, 't' for minutely, 's' for secondly)
- `--target`: Target variable to predict
- `--embed`: Type of embedding to use ('timeF', 'fixed', 'learned')
- `--percent`: Percentage of data to use for training (e.g., 10, 20, 50, 100)

## Model Architecture Parameters

### Transformer Configuration
- `--d_model`: Dimension of the model's hidden states
- `--n_heads`: Number of attention heads
- `--d_ff`: Dimension of the feed-forward network (defaults to 4 * d_model)
- `--e_layers`: Number of encoder layers
- `--d_layers`: Number of decoder layers
- `--d_llm`: Dimension of the LLM embeddings for multimodal fusion
- `--text_emb`: Dimension of text embeddings
- `--llm_dim`: Dimension of LLM output
- `--llm_layers`: Number of LLM layers to use
- `--llm_model`: LLM model to use ('LLAMA2', 'Doc2Vec', etc.)

### Sequence Configuration
- `--seq_len`: Length of input sequences
- `--pred_len`: Length of prediction horizon
- `--enc_in`: Number of input variables
- `--dec_in`: Number of decoder input variables
- `--c_out`: Number of output variables
- `--text_len`: Maximum length of text sequences

### Training Parameters
- `--train_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--patience`: Number of epochs to wait before early stopping
- `--learning_rate`: Learning rate for model training
- `--learning_rate2`: Learning rate for MLP training
- `--learning_rate3`: Learning rate for projection layer training
- `--learning_rate_weight`: Learning rate for weight parameters
- `--des`: Description of the experiment
- `--loss`: Loss function to use ('mse', 'mae', 'mape', 'smape', 'mase')
- `--lradj`: Learning rate adjustment strategy ('type1', 'type2')
- `--use_amp`: Whether to use automatic mixed precision (0 or 1)

### GPU Configuration
- `--use_gpu`: Whether to use GPU (0 or 1)
- `--gpu`: GPU device ID
- `--use_multi_gpu`: Whether to use multiple GPUs (0 or 1)
- `--devices`: Comma-separated list of GPU device IDs for multi-GPU training

### Multimodal Fusion Parameters
- `--prompt_weight`: Weight for prompt-based fusion
- `--type_tag`: Type of tag to use for fusion
- `--pool_type`: Type of pooling for text embeddings ('mean', 'max', 'min', 'attention')
- `--use_fullmodel`: Whether to use the full model for fusion (0 or 1)
- `--huggingface_token`: HuggingFace API token for model access

## Model-Specific Details

### iTransformer Model
The iTransformer model uses a linear projection head for different tasks:
- For forecasting tasks: Projects from d_model to pred_len
- For imputation: Projects from d_model to seq_len
- For anomaly detection: Projects from d_model to seq_len
- For classification: Projects from d_model * enc_in to num_class

The model includes:
1. Data embedding layer (positional, token, or temporal)
2. Encoder with multiple attention layers
3. Task-specific projection head

### Loss Functions
Available loss functions:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

## Example Usage

### Long-term Forecasting
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model iTransformer \
    --data ETTh1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1.csv \
    --freq h \
    --seq_len 96 \
    --pred_len 24 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 2048 \
    --batch_size 32 \
    --train_epochs 10
```

### Multimodal Fusion
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model iTransformer \
    --data ETTh1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1.csv \
    --text_path text_data.csv \
    --freq h \
    --seq_len 96 \
    --pred_len 24 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 2048 \
    --d_llm 768 \
    --text_emb 256 \
    --pool_type attention \
    --prompt_weight 0.5
```

## Notes
- The model architecture and parameters should be chosen based on the specific task and dataset characteristics
- For multimodal fusion, ensure that both time series and text data are properly aligned and preprocessed
- GPU memory requirements vary based on model size and batch size
- Early stopping is implemented to prevent overfitting
- Learning rate adjustment strategies can help optimize training 