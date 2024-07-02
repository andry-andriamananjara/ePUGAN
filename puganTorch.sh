#!/bin/bash

#SBATCH --job-name=PU1K_non_uniform
#SBATCH --account=project_2009906
#SBATCH --partition=gpu
##SBATCH --time=2-00:00:00
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"

cd train
##srun python train.py --exp_name=PU1K_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k
srun python train.py --exp_name=PU1K_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k 