#!/bin/bash

#SBATCH --job-name=Measurement
#SBATCH --account=project_2009906
#SBATCH --partition=gpu
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"

cd test
### DataSet PUGAN
##srun python pc_upsampling.py --non_uniform False --resume=../checkpoints/trainPUGAN_non_uniform/G_iter_99.pth --path=../Mydataset/PUGAN/pugan_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform True --resume=../checkpoints/trainPUGAN_non_uniform/G_iter_99.pth --path=../Mydataset/PUGAN/pugan_non_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform False --resume=../checkpoints/trainPUGAN_uniform/G_iter_99.pth --path=../Mydataset/PUGAN/pugan_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform True --resume=../checkpoints/trainPUGAN_uniform/G_iter_99.pth --path=../Mydataset/PUGAN/pugan_non_uniform_2048_8192_test.h5

#### DataSet PU1K

##srun python pc_upsampling.py --non_uniform False --resume=../checkpoints/PU1K_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform True --resume=../checkpoints/PU1K_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform False --resume=../checkpoints/PU1K_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --non_uniform True --resume=../checkpoints/PU1K_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba                            --non_uniform False --resume=../checkpoints/trainpu1kGen_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba                            --non_uniform True  --resume=../checkpoints/trainpu1kGen_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba                              --non_uniform False --resume=../checkpoints/trainpu1kGen_non_unif/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba                              --non_uniform True  --resume=../checkpoints/trainpu1kGen_non_unif/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

## srun python pc_upsampling.py                          --dis_attention=mamba   --non_uniform False --resume=../checkpoints/trainpu1kDis_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
## srun python pc_upsampling.py                          --dis_attention=mamba   --non_uniform True  --resume=../checkpoints/trainpu1kDis_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

## srun python pc_upsampling.py                        --dis_attention=mamba   --non_uniform False --resume=../checkpoints/trainpu1kDis_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
## srun python pc_upsampling.py                        --dis_attention=mamba   --non_uniform True  --resume=../checkpoints/trainpu1kDis_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba      --non_uniform False --resume=../checkpoints/trainpu1kGenDis_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba      --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba        --non_uniform False --resume=../checkpoints/trainpu1kGenDis_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##(24,16,4) srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba        --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5


## Mamba default value

##srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba --non_uniform False --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba --non_uniform False --resume=../checkpoints/trainpu1kGenDis_default_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py --gen_attention=mamba --dis_attention=mamba --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_default_non_uniform/G_iter_99.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##P3DConv
##srun python pc_upsampling.py   --non_uniform False --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_uniform/G_iter_89.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py   --non_uniform True --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_uniform/G_iter_89.pth     --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##srun python pc_upsampling.py   --non_uniform False --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_non_uniform/G_iter_89.pth --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
##srun python pc_upsampling.py   --non_uniform True --feat_ext=P3DConv --resume=../checkpoints/trainpu1kP3Dconv_non_uniform/G_iter_89.pth     --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

##P3DConv + mamba2
srun python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
srun python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

srun python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_non_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
srun python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_non_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

srun python pc_upsampling.py --gen_attention=mamba2                                          --non_uniform False  --resume=../checkpoints/trainpu1kGen2_uniform/G_iter_99.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
srun python pc_upsampling.py --gen_attention=mamba2                                          --non_uniform True   --resume=../checkpoints/trainpu1kGen2_uniform/G_iter_99.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5

srun python pc_upsampling.py --gen_attention=mamba2                                          --non_uniform False  --resume=../checkpoints/trainpu1kGen2_non_uniform/G_iter_99.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
srun python pc_upsampling.py --gen_attention=mamba2                                          --non_uniform True   --resume=../checkpoints/trainpu1kGen2_non_uniform/G_iter_99.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5


