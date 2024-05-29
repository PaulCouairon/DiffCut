#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=1200

eval "$(conda shell.bash hook)"
conda activate diffseg

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"

python eval_segmentation_diffcut.py --model_name SSD-1B --tau 0.5 --dataset_name Cityscapes --alpha 10 --refinement

