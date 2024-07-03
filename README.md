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

## Installation
<!-- Installation -->
After exporting the environment, install **requirements.txt** and **update.txt** :
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

If settings need to be adjusted, such as changing the data location, parameter values, or checkpoint paths, manual changes can be made in the folder **option**. This folder contains all the default configurations for both training and testing sessions.

## Dataset
<!-- Dataset -->
During the training, we use [PU-NET](https://github.com/yulequan/PU-Net) and  [PU1K](https://github.com/guochengqian/PU-GCN) datasets, and they can be also found in `MC_5k/Mydataset`.

Given the necessity for training with both uniform and non-uniform datasets, the current availability is limited to non-uniform datasets with fixed input. Moreover, there are issues with the existing code used to generate datasets from mesh. Below, we introduce a new mesh data generator code aimed at creating versatile input datasets, whether uniform or non-uniform.
```
XXXXXX
XXXXXX
XXXXXX
```

<!-- Creating new dataset from mesh file -->

## Training
<!-- Run Training -->
```
cd train
python train.py --exp_name=PU1K_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=nameofdataset 
```
By default, the input is non-uniform. If a uniform input is needed, simply add --uniform to the command line. The available datasets are `punet`, `pugan`, and `pu1k`. For more details, refer to `data/data_loader.py`.

```
cd train
python train.py --exp_name=anyexperiencename     --gpu=1 --use_gan --batch_size=12 --uniform --dataname=nameofdataset
```

## Testing
<!-- Run Testing -->

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
