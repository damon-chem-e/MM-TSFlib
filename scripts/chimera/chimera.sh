export CUDA_VISIBLE_DEVICES=0

root_paths=("./data/Public_Health")
data_paths=("US_FLURATIO_Week.csv") 
pred_lengths=(12 24 36 48)
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
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${model_id}_${prompt_weight}_${pred_len} \
        --model ChimeraTransformer \
        --data custom \
        --features M \
        --seq_len 24 \
        --label_len 12 \
        --pred_len $pred_len \
        --des 'Exp' \
        --seed $seed \
        --type_tag "#F#" \
        --text_len 4 \
        --prompt_weight 0.3 \
        --pool_type "avg" \
        --save_name "results/result_chimera_test.txt" \
        --llm_model GPT2 \
        --huggingface_token 'NA'\
        --use_fullmodel $use_fullmodel
    done
  done
done

