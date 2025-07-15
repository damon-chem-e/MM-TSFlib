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

# Read all categories from the text file
while IFS= read -r category; do
    # Skip empty lines
    if [[ -z "$category" ]]; then
        continue
    fi
    
    echo "Downloading data for category: $category"
    
    python \
           /home/damonp/projects/chimera_proj/amazon-data/lib/dataset_builder.py --slurm \
           download-huggingface \
           --category "$category" \
           --work-dir /home/damonp/projects/chimera_proj/amazon-data
    
    echo "Completed download for category: $category"
    echo "----------------------------------------"
    
done < all_categories.txt

echo "All category downloads completed!" 