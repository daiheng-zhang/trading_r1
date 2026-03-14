cd /home1/10499/daiheng367/trading_r1
source /scratch/10499/daiheng367/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10499/daiheng367/miniconda3/envs/trading_r1

PROJECT_DIR=/home1/10499/daiheng367/trading_r1
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-trading_r1}"


python __main__.py train-sft --config configs/train_stage2_sft.yaml