# ePUGAN based on mamba, P3DNet, and arbitrary-scale for point cloud upsampling
The code in this repository focuses on the Enhanced Point Cloud Upsampling GAN (ePUGAN), a modified version of PU-GAN that incorporates recent advancements in deep learning, including MAMBA, P3DNet, and Arbitrary-Scale techniques. This repository is updated progressively.

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
Load purge and tykky, modules of CSC :
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

If you have additional packages to install, create a new file e.g. update.txt and inside the file, write the package as follow
```
pip install package_name (if it is from pip)
or
conda install pytorch3d -c pytorch3d -y (if it is from conda)
```

then run the command
```
conda-containerize update /projappl/project_2009906/Env/venv_3dpytorch --post-install update.txt
```

## Additional configurations
<!-- Additional configurations -->

<!-- Dataset -->

<!-- New dataset -->

<!-- Creating new dataset from mesh file -->

<!-- Run Training -->

<!-- Run Testing -->

