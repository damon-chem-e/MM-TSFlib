export CUDA_VISIBLE_DEVICES=0

root_paths=("./data/Public_Health")
data_paths=("US_FLURATIO_Week.csv") 
pred_lengths=(12 24 36 48)
layers=(2 4)
seeds=(2021)
use_fullmodel=0
length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for ((i=0; i<$length; i++))
  do
    for pred_len in "${pred_lengths[@]}"
    do
      for num_layers in "${layers[@]}"
      do 
        for n_heads in {4..12}
        do
          root_path=${root_paths[$i]}
          data_path=${data_paths[$i]}
          model_id=$(basename ${root_path})  
          echo "Running ChimeraTransformer with pred_len $pred_len n_heads $n_heads layers $num_layers"
          python -u run.py \
            --model ChimeraTransformer \
            --ts_only 1 \
            --task_name long_term_forecast \
            --is_training 1 \
            --num_layers $num_layers \
            --n_heads $n_heads \
            --root_path $root_path \
            --data_path $data_path \
            --model_id pred${pred_len}_heads${n_heads}_layers${num_layers} \
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
            --save_name "results/grid_heads_layers.txt" \
            --llm_model GPT2 \
            --huggingface_token 'NA'\
            --use_fullmodel $use_fullmodel
        done
      done
    done
  done
done

