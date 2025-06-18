#!/bin/bash
# Script to allocate resources

# salloc -N 1 -n 4 --gres=gpu:a100:1 -p sched_mit_sloan_gpu_r8 -t "${1:-120}"
salloc -N 1 -n 4 --gres=gpu:h200:1 -p mit_preemptable -t "${1:-120}"
module load miniforge
source activate chimera
uv sync
source .venv/bin/activate