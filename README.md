# ePUGAN based on mamba, P3DNet, and arbitrary-scale for point cloud upsampling
The code in this repository focuses on the Enhanced Point Cloud Upsampling GAN (ePUGAN), a modified version of PU-GAN that incorporates recent advancements in deep learning, including MAMBA, P3DNet, and Arbitrary-Scale techniques. This repository is updated progressively.

<!-- Environment -->
## Environment
<!-- This section is only for CSC environment, you can skip if you are not using CSC supercomputer-->

This section is only for CSC environment, you can skip if you are not using CSC supercomputer.

```
cd /projappl/project_2009906
mkdir Env
cd Env
mkdir venv_3dpytorch


module purge
module load tykky
conda-containerize new --prefix venv_3dpytorch pugan_torch.yaml
```


run python command directly :
```
export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"
```

run python python command on top of GPU allocation :
```
sinteractive -j anyrandom_name -A project_2009906 -t 10:00:00 -m 64G -c 8 -p gpu -g 1
export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"
```


<!-- Installation -->

<!-- Additional configurations -->

<!-- Dataset -->

<!-- New dataset -->

<!-- Creating new dataset from mesh file -->

<!-- Run Training -->

<!-- Run Testing -->

