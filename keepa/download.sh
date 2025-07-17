#!/bin/bash
#SBATCH -n 16 #Request 16 tasks (cores)
#SBATCH -N 1 #Request 1 nodes
#SBATCH -t 0-06:00:00 #Request runtime of 6 hrs (longer for all categories)
#SBATCH -p mit_normal  # partition choice
#SBATCH --mem-per-cpu 4000 #Request 4G of memory per CPU (64 gb total)
#SBATCH -o /home/damonp/projects/chimera_proj/amazon-data/logs/download_all.log #redirect output to output_JOBID.txt
#SBATCH -e /home/damonp/projects/chimera_proj/amazon-data/logs/download_all.err #redirect errors to error_JOBID.txt

source /home/damonp/.bashrc
source /home/damonp/venvs/amazon-data/bin/activate

# Call the orchestrator script with download command and SLURM mode
python orchestrator.py download 