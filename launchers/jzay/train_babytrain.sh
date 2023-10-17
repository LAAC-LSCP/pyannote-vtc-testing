#!/bin/bash
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=xdz@v100

python main.py runs/babytrain_BBT2_42/ train \
-p X.SpeakerDiarization.BBT2 \
--classes babytrain \
--model_type pyannet \
--epoch 100
