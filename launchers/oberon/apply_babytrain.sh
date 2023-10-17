#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py runs/babytrain/ apply \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/babytrain/checkpoints/last.ckpt \
--classes babytrain \
--apply_folder runs/babytrain/apply/ \
--params runs/babytrain/best_params.yml
