# Introduction
This repository contains code necessary to not only reproduce the experiments presented in [Scalable Gaussian Process Regression with Derivatives @ NIPS 2018](https://arxiv.org/abs/1810.12283), but also for any
wishing to use our methods.

Directory Structure: our root directory contains the `code/` and `demos/` folders, which contain the research code written for
our paper and the demo code to reproduce our results, respectively. `demos/` is contains a number of subdirectories, each
corresponding to an experiment run in our paper, and we will detail these later on in the readme. `code/` has the following subdirectories:

* `bayesopt/`
* `estimator/`
* `gp/`
* `kernels/`
* `lik/`
* `mvm/`
* `testfunctions/`
* `utils/`

`bayesopt/` contains code for our Bayesian optimization experiments. `estimator/` contains code for stochastic trace estimation detailed
in [this](https://arxiv.org/pdf/1711.03481.pdf) paper. `gp/` is the primary GPML-like interface for training a Gaussian process. `kernels/` 
contains implementations of various kernels, including the D-SKI and D-SKIP kernels we developed. `lik/` contains standard marginal log likelihood
functions and `mvm/` contains fast matrix-vector multiplication subroutines. `testfunctions/` contains standard global optimization test functions 
used to test our Bayesian optimization routines. Lastly, `utils/` contains various other subroutines we use. 

# Set-up
We use Matlab's new element-wise multiplication operators, and thus require Matlab R2017A at a minimum. 
To start, clone the repository and [mex](https://www.mathworks.com/help/matlab/ref/mex.html) `ski_diag.c`, a 
key routine extracting the diagonal of the D-SKI matrix needed when preconditioning. Mexing allows one to run c code 
from matlab, but mexing `ski_diag.c` is not necessary if you do not plan on using D-SKI. 
Before attempting any experiments, make sure to run `startup.m`, which
simply adds all necessary directories and subdirectories to Matlab's path. 

# Training a GP
To train a basic GP with no gradients, run:

`[mu, K] = gp(X, Y)`

where the pair `X,Y` represent the training points and values respectively. `mu` is a function handle predicting on new points e.g. `y_predict = mu(x_predict)`, and `K` is the dense
kernel matrix. The same idea holds when gradients are available

`[mu, K] = gp_grad(X, Y, dY)`

where `dY` contains the respective gradients of points in `X`. Similar interfaces are available for (D-)SKI and (D-)SKIP kernels, except `K` is
a function handle performing matrix-vector multiplication instead of a dense matrix. 


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
lengthscale and regularization parameters. Run `PrecondCompare.m` located in `/Demos/Precond` to recreate the following plot (figure 8 in the appendix).

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
points and gradients, reconstruction will take a bit of time. Run `bunny.m` located in `/Demos/ImplicitBunny` to recreate the following plot (figure 8 in the appendix).

<p align="center">
    <img src="/images/Bunny.png" width="1000">
</p>

# Bayesian Optimization Experiment

Dimensionality reduction with gradient information is possible estimating the dominant active subspace. We imbed two popular test functions for global
optimization, the Ackley and Rastragin functions, in higher dimensions and run scalable Bayesian optimization with dimensionality reduction and see
promising results. Run `boComparison.m` located in `/Demos/BayesOpt` to recreate the following plot (figure 8 in the appendix).

<p align="center">
    <img src="/images/ackley.png" width="400">
    <img src="/images/rastrigin.png" width="400">
</p>
