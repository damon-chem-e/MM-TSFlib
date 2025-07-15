#!/bin/bash
#SBATCH -n 16 #Request 16 tasks (cores)
#SBATCH -N 1 #Request 1 nodes
#SBATCH -t 0-03:00:00 #Request runtime of 3 hrs per job
#SBATCH -p mit_normal  # partition choice
#SBATCH --mem-per-cpu 4000 #Request 4G of memory per CPU (64 gb total)
#SBATCH --array=0-33 #Job array for 34 categories (0-indexed)
#SBATCH -o /home/damonp/projects/chimera_proj/amazon-data/logs/processing_%A_%a.log #redirect output with job array info
#SBATCH -e /home/damonp/projects/chimera_proj/amazon-data/logs/processing_%A_%a.err #redirect errors with job array info

source /home/damonp/.bashrc
source /home/damonp/venvs/amazon-data/bin/activate

# Read the category corresponding to this array job
mapfile -t categories < all_categories.txt
category="${categories[$SLURM_ARRAY_TASK_ID]}"

echo "Processing category: $category (Array job $SLURM_ARRAY_TASK_ID)"

# Process with calendar windowing strategy
echo "Processing calendar windowing strategy for $category"
python \
       /home/damonp/projects/chimera_proj/amazon-data/lib/dataset_builder.py --slurm \
       process-huggingface-only \
       --category "$category" \
       --windowing-strategy calendar \
       --calendar-window-interval 1w \
       --upsample \
       --work-dir /home/damonp/projects/chimera_proj/amazon-data \
       --sub-dir calendar_1w \
       --polars

# Process with review frequency windowing strategy
echo "Processing review frequency windowing strategy for $category"
python \
       /home/damonp/projects/chimera_proj/amazon-data/lib/dataset_builder.py --slurm \
       process-huggingface-only \
       --category "$category" \
       --windowing-strategy review_frequency \
       --review-window-size 10 \
       --work-dir /home/damonp/projects/chimera_proj/amazon-data \
       --sub-dir review_frequency_10 \
       --polars

echo "Completed processing for category: $category"
