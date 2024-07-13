# ePUGAN based on MAMBA, P3DNet, and arbitrary-scale for point cloud upsampling
The code in this repository focuses on the Enhanced Point Cloud Upsampling GAN (ePUGAN) model, a modified version of PU-GAN that incorporates recent advancements in deep learning, including MAMBA, P3DNet, and Arbitrary-Scale techniques. Additionally, all C++ dependencies have been replaced with `PyTorch3D` code. This repository is updated progressively.

<!-- Environment -->
## Environment

This section is only relevant for the CSC environment. If you are not using the CSC supercomputer, please proceed directly to the next section, **installation**.

Create folders **Env** and **venv_3dpytorch** :
```
cd /projappl/project_2009906
mkdir Env
cd Env
mkdir venv_3dpytorch
```
Load purge and tykky, modules of CSC, then build the environment from **pugan_torch.yaml** :
```
module purge
module load tykky
conda-containerize new --prefix venv_3dpytorch pugan_torch.yaml
```

Export the environment as shown below to enable running Python commands:
```
export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"
```

Apply for GPU resource in CSC using `sinteractive` by following the general rule below
```
sinteractive -j anysessionname -A project_ID_of_yourproject -t hh:mm:ss -m 64G -c nb_core -p gpu -g nb_gpu
```
example
```
sinteractive -j anysessionname -A project_2009906 -t 10:00:00 -m 64G -c 8 -p gpu -g 4
or
sinteractive -j anysessionname -A project_2009906 -t 10:00:00 -m 64G -c 8 -p gpu -g 1
```

## Installation
<!-- Installation -->
After exporting the environment, install **requirements.txt** and **req_pytorch3d.txt** :
```
conda-containerize update /projappl/project_2009906/Env/venv_3dpytorch --post-install requirements.txt
conda-containerize update /projappl/project_2009906/Env/venv_3dpytorch --post-install req_pytorch3d.txt
```

If you have additional packages to install, create a new file e.g. update.txt and inside the file, write the package's name as follow
```
pip install package_name -y
or
conda install -n myenv package_name -y
```

then run the command
```
conda-containerize update /projappl/project_2009906/Env/venv_3dpytorch --post-install update.txt
```

## Additional configurations
<!-- Additional configurations -->

If settings need to be adjusted, such as changing the data location, parameter values, or checkpoint paths, manual changes can be made in files `option/train_option.py` and `option/test_option.py`. This folder contains all the default configurations for both training and testing sessions.

## Dataset
<!-- Dataset -->
***The code structure in this section will be updated in the future for improved usability.**

During the training, we use [PU-NET](https://github.com/yulequan/PU-Net) and  [PU1K](https://github.com/guochengqian/PU-GCN) datasets, and they can be also found in `MC_5k/Mydataset`.

Given the necessity for training with both uniform and non-uniform datasets, the current availability is limited to non-uniform datasets with fixed input. Moreover, there are issues with the existing code used to generate datasets from mesh. Below, we introduce a new mesh data generator code aimed at creating versatile input datasets, whether uniform or non-uniform. Further details on number of patches and patch size can be seen 
in `MC_5K/off_to_xyz.py`.

```
cd MC_5K
python off_to_xyz.py --isTrain=train                   --datasetdir=Mydataset/PU1K/non_uniform/trainpu1k
python off_to_xyz.py --isTrain=test                    --datasetdir=Mydataset/PU1K/non_uniform/test/original_meshes
```

If you have many separate .xyz and .pcd files that are large and need to be cropped, follow examples below.
```
cd MC_5K
python crop_point_cloud.py --isTrain=test --datasetdir=Mydataset/PU_LiDAR
python crop_point_cloud.py --isTrain=test --datasetdir=Mydataset/PU_LiDAR/rectified_scans_local
```
<!-- Creating new dataset from mesh file -->

## Training
<!-- Run Training -->
```
cd train
python train.py --exp_name=PU1K_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=nameofdataset 
```
By default, the input is non-uniform. If a uniform input is needed, simply add `--uniform` to the command line. The available datasets are `punet`, `pugan`, and `pu1k`. For more details of dataset, refer to `data/data_loader.py`.

Below is an example of how to train the ePUGAN model using the `pu1k` dataset with uniform input. **If no specifications are provided for the feature extraction, generator, and discriminator, the model defaults to the original PU-GAN**.
```
cd train
python train.py --exp_name=anyexperiencename     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k
```
Below is an example of how to train the ePUGAN model using the `pu1k` dataset with uniform and non-uniform inputs, MAMBA in generator, and P3DNet as Feature Extraction.
```
cd train
python train.py --exp_name=anyexperiencename1     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv
python train.py --exp_name=anyexperiencename2     --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv
```
For more details of the training specification, please refer to the bash file `puganMamba.sh`

## Testing
<!-- Run Testing -->
***The code structure in this section will be updated in the future for improved usability.**

If there is a specific chekpoint to use for testing uniform or non-uniform inputs, follow below expample. Here we use `G_iter_89.pth`. Further details can be found in the bash file `cd_hd_p2f.sh`.
```
cd test
python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
python pc_upsampling.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5
```

If all chekpoints are needed to use for testing uniform or non-uniform inputs, follow below expample. Further details can be found in the bash file `cd_hd_p2f_full_epochs.sh`.
```
cd test
python pc_upsampling_all_epochs.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform False  --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_uniform_2048_8192_test.h5
python pc_upsampling_all_epochs.py --gen_attention=mamba2                       --feat_ext=P3DConv --non_uniform True   --resume=../checkpoints/trainpu1kGen2_P3Dconv_uniform/G_iter_89.pth   --path=../MC_5k/Mydataset/PU1K/non_uniform/test/pu1k_non_uniform_2048_8192_test.h5
```

If testing and visualization with an xyz file is needed, follow the example below. Further details can be found in the bash file `predict_testxyz_viz.sh`.
```
cd test
python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform False --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
python pc_upsampling_xyz.py --gen_attention=mamba --dis_attention=mamba --non_uniform True  --resume=../checkpoints/trainpu1kGenDis_default_uniform/G_iter_99.pth --path=--path=../MC_5k/Mydataset/PU_LiDAR/xyz_file
```

## References
```
@InProceedings{He_2023_CVPR,
    author    = {He, Yun and Tang, Danhang and Zhang, Yinda and Xue, Xiangyang and Fu, Yanwei},
    title     = {Grad-PU: Arbitrary-Scale Point Cloud Upsampling via Gradient Descent with Learned Distance Functions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}

@InProceedings{Qian_2021_CVPR,
    author    = {Qian, Guocheng and Abualshour, Abdulellah and Li, Guohao and Thabet, Ali and Ghanem, Bernard},
    title     = {PU-GCN: Point Cloud Upsampling Using Graph Convolutional Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11683-11692}
}

@inproceedings{li2019pugan,
     title={PU-GAN: a Point Cloud Upsampling Adversarial Network},
     author={Li, Ruihui and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
     booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
     year = {2019}
 }

@inproceedings{yu2018pu,
     title={PU-Net: Point Cloud Upsampling Network},
     author={Yu, Lequan and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
     booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
     year = {2018}
}
```
