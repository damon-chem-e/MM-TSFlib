export CUDA_VISIBLE_DEVICES=0

root_paths=("./data/Public_Health")
data_paths=("US_FLURATIO_Week.csv") 
# pred_lengths=(12 24 36 48)
pred_lengths=(12)
seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for ((i=0; i<$length; i++))
  do
    for pred_len in "${pred_lengths[@]}"
    do
      root_path=${root_paths[$i]}
      data_path=${data_paths[$i]}
      model_id=$(basename ${root_path})  
      echo "Running model ChimeraTransformer with pred_len $pred_len"
      python -u run.py \
        --model ChimeraTransformer \
        --ts_only 0 \
        --load_ts 0 \
        --freeze_ts 0 \
        --task_name long_term_forecast \
        --is_training 1 \
        --gate_type linear_norm \
        --gate_regularization_lambda 0.0 \
        --n_heads 12 \
        --d_latent 516 \
        --num_layers 4 \
        --num_layers_llm 4 \
        --fusion_heads 12 \
        --post_fusion_layers 4 \
        --final_layers 4 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${model_id}_${pred_len} \
        --train_epochs 10 \
        --data custom \
        --features M \
        --seq_len 24 \
        --label_len 12 \
        --pred_len $pred_len \
        --des 'Exp' \
        --seed $seed \
        --type_tag "#F#" \
        --text_len 4 \
        --save_name "results/result_chimera_test.txt" \
        --llm_model GPT2 \
        --huggingface_token 'NA'\
        --use_fullmodel $use_fullmodel
    done
  done
done

