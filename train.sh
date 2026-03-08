#!/bin/bash
#SBATCH -J trading_r1-grpo
#SBATCH -o log/grpo_%j.log
#SBATCH -e log/grpo_%j.log
#SBATCH -A Deep-Learning-at-Sca
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 48:00:00
#SBATCH --exclude=c561-007,c563-001,c561-001

cd /home1/10499/daiheng367/trading_r1
source /scratch/10499/daiheng367/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10499/daiheng367/miniconda3/envs/trading_r1
0
PROJECT_DIR=/home1/10499/daiheng367/trading_r1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-trading_r1}"


torchrun --nproc_per_node=4 __main__.py train-grpo --config configs/train_stage1_grpo.yaml