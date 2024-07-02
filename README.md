# ePUGAN based on mamba, P3DNet, and arbitrary-scale for point cloud upsampling
The code in this repository focuses on the Enhanced Point Cloud Upsampling GAN (ePUGAN), a modified version of PU-GAN that incorporates recent advancements in deep learning, including MAMBA, P3DNet, and Arbitrary-Scale techniques. This repository is updated progressively.

<!-- Environment -->
## Environment

This section is only relevant for the CSC environment. If you are not using the CSC supercomputer, please proceed directly to the next section, **installation**.

```
cd /projappl/project_2009906
mkdir Env
cd Env
mkdir venv_3dpytorch


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

<!-- Additional configurations -->

<!-- Dataset -->

<!-- New dataset -->

<!-- Creating new dataset from mesh file -->

<!-- Run Training -->

<!-- Run Testing -->

