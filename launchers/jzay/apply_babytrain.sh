#!/bin/bash
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
##SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=xdz@v100


python main.py runs/babytrain_BBT2_1/ apply \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/babytrain_BBT2_1/checkpoints/best.ckpt \
--classes babytrain \
--apply_folder runs/babytrain_BBT2_1/apply/ \
--params runs/babytrain_BBT2_1/best_params.yml
