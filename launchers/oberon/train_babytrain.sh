#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py runs/babytrain/ train \
-p X.SpeakerDiarization.BBT2 \
--classes babytrain \
--model_type pyannet \
--epoch 100
