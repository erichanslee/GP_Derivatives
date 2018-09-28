# Introduction
This repository contains code necessary to not only reproduce the experiments presented in [], but also for any
wishing to use our methods.

# Set-up
We use Matlab's new element-wise multiplication operators, and thus require Matlab R2017A at a minimum. 
To start, clone the repository and [mex](https://www.mathworks.com/help/matlab/ref/mex.html) `ski_diag.c`, a 
key routine extracting the diagonal of the D-SKI matrix needed when preconditioning. Then run `startup.m`, which
simply adds all directories and subdirectories to Matlab's path. 

# Easily using D-SKI and D-SKIP


# Spectrum Comparison Experiment
Any approximate kernel's spectrum should closely match the spectrum of the original kernel. This experiment
shows that our approximate kernels D-SKI and D-SKIP indeed possess this property. Run `SpectrumError.m` 
located in `/Demos/Spectrum/` to generate the following plot (figure 2 of section 4). 

<p align="center">
    <img src="/images/SpectrumError.png" width="700">
</p>


# MVM Scaling Experiment
The cost of an matrix-vector multiply with a dense kernel matrix is quadratic, whereas the cost of a matrix-vector multiply
with either D-SKI or D-SKIP is linear, which is verified in this experiment. Run `ScalingComparison.m` 
located in `/Demos/Scaling/` to generate the following plot (figure 3 of section 4). 

<p align="center">
    <img src="/images/ScalingComparison.png" width="700">
</p>


# Dimensionality Reduction Experiment
Possessing gradients allows one to perform dimensionality reduction via active subspace sampling. Run `DimReduction.m` 
located in `/Demos/ActiveSubspace` to generate the following plot (figure 4 and table 2 of section 4). For quick runtimes, the
demo code uses far fewer points than the experiment presented in the paper. The conclusion remains the same; D-SKIP with active
subspace sampling performs far better than the dense code. 

<p align="center">
    <img src="/images/DimRed1.png" width="200">
    <img src="/images/DimRed2.png" width="200">
    <img src="/images/DimRed3.png" width="200">
    <img src="/images/DimRed4.png" width="200">
</p>


# Preconditioning Experiment
Preconditioning is key to increasing convergence of iterative methods; we found that pivoted Cholesky is an effective preconditioner.
In general, convergence of iterative methods depends on the spectrum of the kernel matrix, and if there is spectral clustering or a large spectral
gap, we expect fast convergence. In this experiment, we consider the SE Kernel and plot the effectiveness of pivoted Cholesky in terms of both
lengthscale and regularization parameters. Run `PrecondCompare` located in `/Demos/Precond` to recreate the following plot (figure 8 in the appendix).

<p align="center">
    <img src="/images/Precond.png" width="700">
</p>

# Rough Terrain Reconstruction Experiments
We apply our method to rough terrain reconsutrction of Mount St. Helens from lidar 
[data](https://wagda.lib.washington.edu/data/type/elevation/lidar/st_helens/). 
Run `MountStHelens.m` located in `/Demos/Maps` to recreate the following plot (figure 5 in section 4).

<p align="center">
    <img src="/images/MountStHelens.png" width="1000">
</p>

# Implicit Surface Reconstruction Experiment
We apply our method to implicit surface reconstruction. We reconstruct the [Stanford bunny](http://graphics.stanford.edu/data/3Dscanrep/) 
from noisy surface normals (i.e. gradients), which is difficult to do using traditional spline methods. Note that, as we are using roughly 40 thousand
points and gradients, reconstruction will take a bit of time. 

<p align="center">
    <img src="/images/Bunny.png" width="1000">
</p>

# Bayesian Optimization Experiment

<p align="center">
    <img src="/images/ackley.png" width="400">
    <img src="/images/rastrigin.png" width="400">
</p>