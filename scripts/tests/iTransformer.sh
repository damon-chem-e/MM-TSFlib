export CUDA_VISIBLE_DEVICES=0

root_paths=("./data/Public_Health")
data_paths=("US_FLURATIO_Week.csv") 
pred_lengths=(12 24 36 48)
seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for prompt_weight in $(seq 0.1 0.05 0.8)
  do
    for ((i=0; i<$length; i++))
    do
      for pred_len in "${pred_lengths[@]}"
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})

        echo "Running model iTransformer with prompt_weight $prompt_weight, pred_len $pred_len"
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id ${model_id}_${prompt_weight}_${pred_len} \
          --model iTransformer \
          --data custom \
          --features M \
          --seq_len 24 \
          --label_len 12 \
          --pred_len $pred_len \
          --des 'Exp' \
          --seed $seed \
          --type_tag "#F#" \
          --text_len 4 \
          --prompt_weight $prompt_weight \
          --pool_type "avg" \
          --save_name "results/result_iTrans_prompt_weight_optim.txt" \
          --llm_model GPT2 \
          --huggingface_token 'NA'\
          --use_fullmodel $use_fullmodel
      done
    done
  done
done

