#!/bin/bash

#SBATCH --job-name=trainpu1kGen2Dis2_non_uniform
#SBATCH --account=project_2009906
#SBATCH --partition=gpu
##SBATCH --time=2-00:00:00
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"

cd train
##srun python train.py --exp_name=trainpu1kGen_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba
##srun python train.py --exp_name=trainpu1kGen_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba

##srun python train.py --exp_name=trainpu1kDis_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kDis_non_uniform --gpu=1 --use_gan --batch_size=12       --dataname=pu1k --dis_attention=mamba

##srun python train.py --exp_name=trainpu1kGenDis_uniform     --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kGenDis_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba

## Mamba default value (16,4,2)
##srun python train.py --exp_name=trainpu1kGen1642_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba
##srun python train.py --exp_name=trainpu1kGen1642_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba

##srun python train.py --exp_name=trainpu1kGenDis_default_uniform     --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kGenDis_default_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba

############################# Feature extraction
##srun python train.py --exp_name=trainpu1kP3Dconv_uniform   --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k    --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kP3Dconv_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --feat_ext=P3DConv

##srun python train.py --exp_name=trainpu1kGenP3Dconv_default_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kGenP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv

##srun python train.py --exp_name=trainpu1kDisP3Dconv_default_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --dis_attention=mamba --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kDisP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12       --dataname=pu1k --dis_attention=mamba --feat_ext=P3DConv

##srun python train.py --exp_name=trainpu1kGenDisP3Dconv_default_uniform     --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kGenDisP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba --feat_ext=P3DConv

############################ mamba2
##srun python train.py --exp_name=trainpu1kGen2_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba2
##srun python train.py --exp_name=trainpu1kGen2_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba2

##srun python train.py --exp_name=trainpu1kGen2Dis2_uniform     --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba2 --dis_attention=mamba2
srun python train.py --exp_name=trainpu1kGen2Dis2_non_uniform      --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba2 --dis_attention=mamba2

##srun python train.py --exp_name=trainpu1kGen2_P3Dconv_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba2 --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kGen2_P3Dconv_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba2 --feat_ext=P3DConv

############################ Arbitrary Scale
##srun python train.py --exp_name=trainpu1kArbScale_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k_scale                                                     --scale=arbitrary
##srun python train.py --exp_name=trainpu1kArbScale_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k_scale                                                      --scale=arbitrary

##srun python train.py --exp_name=trainpu1kGen2_ArbScale_uniform     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k_scale --gen_attention=mamba2                          --scale=arbitrary
##srun python train.py --exp_name=trainpu1kGen2_ArbScale_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k_scale  --gen_attention=mamba2                          --scale=arbitrary
