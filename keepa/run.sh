#!/bin/bash
#SBATCH -n 16 #Request 16 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH -t 0-03:00:00 #Request runtime of 3 hrs per job
#SBATCH -p mit_normal  # partition choice
#SBATCH --mem-per-cpu 4000 #Request 4G of memory per CPU (64 gb total)
#SBATCH --array=0-33 #Job array for 34 categories (0-indexed)
#SBATCH -o /home/damonp/projects/chimera_proj/amazon-data/logs/processing_%A_%a.log #redirect output with job array info
#SBATCH -e /home/damonp/projects/chimera_proj/amazon-data/logs/processing_%A_%a.err #redirect errors with job array info

source /home/damonp/.bashrc
source /home/damonp/venvs/amazon-data/bin/activate

# Simple call to orchestrator - passes array task ID to process single category
python orchestrator.py process-category $SLURM_ARRAY_TASK_ID
