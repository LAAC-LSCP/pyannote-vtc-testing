#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10
#SBATCH --output=replicate_20240327.out

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py apply \
-p X.SpeakerDiarization.BBT2 \
--model_path model_vtc2/checkpoints/best.ckpt \
--classes babytrain \
--apply_folder replicate202403/apply/ \
--params model_vtc2/best_params.yml
