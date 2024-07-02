#!/bin/bash

#SBATCH --job-name=Measurement_epoch_xyz
#SBATCH --account=project_2009906
#SBATCH --partition=gpu
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"

cd test

## PU GAN Default (PUGAN) 

##srun python pc_upsampling_xyz.py --non_uniform False --resume=../checkpoints/trainPUGAN_100K_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU_LiDAR/xyz_file/xyz_file
##srun python pc_upsampling_xyz.py --non_uniform True --resume=../checkpoints/trainPUGAN_100K_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PUGAN/pugan_non_uniform_2048_8192_test.h5
##srun python pc_upsampling_xyz.py --non_uniform False --resume=../checkpoints/trainPUGAN_100K_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU_LiDAR/xyz_file/xyz_file
##srun python pc_upsampling_xyz.py --non_uniform True --resume=../checkpoints/trainPUGAN_100K_niform/G_iter_99.pth --path=../MC_5k/Mydataset/PUGAN/pugan_non_uniform_2048_8192_test.h5


## PU GAN Default (PU1K)
srun python pc_upsampling_xyz.py --non_uniform False --resume=../checkpoints/PU1K_non_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --non_uniform True --resume=../checkpoints/PU1K_non_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --non_uniform False --resume=../checkpoints/PU1K_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --non_uniform True --resume=../checkpoints/PU1K_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

## Dis (16, 4, 2) + mamba v2
srun python pc_upsampling_xyz.py                          --dis_attention=mamba   --non_uniform False --resume=../checkpoints/trainpu1kDis_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py                          --dis_attention=mamba   --non_uniform True  --resume=../checkpoints/trainpu1kDis_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py                        --dis_attention=mamba   --non_uniform False --resume=../checkpoints/trainpu1kDis_non_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py                        --dis_attention=mamba   --non_uniform True  --resume=../checkpoints/trainpu1kDis_non_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file


## Mamba default value
srun python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform False --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform False --resume=../checkpoints/trainpu1kGenDis_default_non_uniform/G_iter_9.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_default_non_uniform/G_iter_9.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

##Mamba2 (Gen2)
srun python pc_upsampling_xyz.py --gen_attention=mamba2                                          --non_uniform False  --resume=../checkpoints/trainpu1kGen2_uniform/G_iter_99.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --gen_attention=mamba2                                          --non_uniform True   --resume=../checkpoints/trainpu1kGen2_uniform/G_iter_99.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py --gen_attention=mamba2                                          --non_uniform False  --resume=../checkpoints/trainpu1kGen2_non_uniform/G_iter_39.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --gen_attention=mamba2                                          --non_uniform True   --resume=../checkpoints/trainpu1kGen2_non_uniform/G_iter_39.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file



##P3DNet
srun python pc_upsampling_xyz.py   --non_uniform False --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_uniform/G_iter_79.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py   --non_uniform True --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_uniform/G_iter_79.pth     --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py   --non_uniform False --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_non_uniform/G_iter_49.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py   --non_uniform True --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_non_uniform/G_iter_49.pth     --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

##P3DConv + mamba2
##srun python pc_upsampling_xyz.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_29.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
##srun python pc_upsampling_xyz.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_39.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

##srun python pc_upsampling_xyz.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_non_uniform/G_iter_19.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
##srun python pc_upsampling_xyz.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_non_uniform/G_iter_69.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

##Arbitrary Scale
srun python pc_upsampling_xyz.py --non_uniform True --scale=arbitrary --resume=../checkpoints/trainpu1kArbScale_non_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --non_uniform False --scale=arbitrary --resume=../checkpoints/trainpu1kArbScale_non_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py --non_uniform True --scale=arbitrary --resume=../checkpoints/trainpu1kArbScale_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py --non_uniform False --scale=arbitrary --resume=../checkpoints/trainpu1kArbScale_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

##Arbitrary Scale + Mamba2

srun python pc_upsampling_xyz.py  --gen_attention=mamba2 --scale=arbitrary                        --non_uniform True  --resume=../checkpoints/trainpu1kGen2_ArbScale_non_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py  --gen_attention=mamba2 --scale=arbitrary                        --non_uniform False --resume=../checkpoints/trainpu1kGen2_ArbScale_non_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file

srun python pc_upsampling_xyz.py  --gen_attention=mamba2 --scale=arbitrary                        --non_uniform True  --resume=../checkpoints/trainpu1kGen2_ArbScale_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
srun python pc_upsampling_xyz.py  --gen_attention=mamba2 --scale=arbitrary                        --non_uniform False --resume=../checkpoints/trainpu1kGen2_ArbScale_uniform/G_iter_49.pth   --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
