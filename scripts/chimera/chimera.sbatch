#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 360
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1

module load miniforge
source activate chimera
uv sync
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

root_paths=("./data/Public_Health")
data_paths=("US_FLURATIO_Week.csv") 
length=${#root_paths[@]}

for seed in {2021...2024}
do
  for ((i=0; i<$length; i++))
  do
    for pred_len in 12 24 36 48
    do
      for ts_only_epoch in 5 10
      do 
        for train_epoch in 5 10
        do
          for option in 0 1
          do
            root_path=${root_paths[$i]}
            data_path=${data_paths[$i]}
            model_id=$(basename ${root_path})  
            echo "Running model ChimeraTransformer with pred_len $pred_len"
            python -u run.py \
              --model ChimeraTransformer \
              --ts_only_epochs $ts_only_epoch \
              --train_epochs $train_epoch \
              --freeze_ts $option \
              --task_name long_term_forecast \
              --is_training 1 \
              --gate_type linear_norm \
              --architecture raw_skip \
              --gate_regularization_lambda 0.0 \
              --n_heads 12 \
              --d_latent 516 \
              --num_layers 4 \
              --num_layers_llm 4 \
              --fusion_heads 12 \
              --post_fusion_layers 4 \
              --final_layers 0 \
              --use_fullmodel 0 \
              --root_path $root_path \
              --data_path $data_path \
              --model_id ${model_id}_${pred_len} \
              --train_epochs 10 \
              --data custom \
              --features M \
              --pred_len $pred_len \
              --des 'Exp' \
              --seed $seed \
              --type_tag "#F#" \
              --text_len 4 \
              --save_name "results/ts_only.txt" \
              --llm_model GPT2 \
              --huggingface_token 'NA'
          done
        done
      done
    done
  done
done