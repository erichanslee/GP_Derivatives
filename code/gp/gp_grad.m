% Interface for training a GP with gradients using the SE Kernel
% 
% [mu, K] = gp_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix

function [mu, K] = gp_grad(X, Y, DY)
[ntrain, d] = size(X);

% Initial hyperparameters
ell0 = 0.5*sqrt(d);
s0 = std(Y);
sig0 = 5e-2*s0; 
beta = 1e-6;

% Train GP 
cov = @(hyp) se_kernel_grad(X, hyp);
lmlfun = @(x) lml_exact(cov, [Y, DY], x, beta);
hyp = struct('cov', log([ell0, s0]), 'lik', log([sig0, sig0]));
params = minimize_quiet(hyp, lmlfun, -50);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE with gradients: (ell, s, sigma1, sigma2) = (%.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Calculate interpolation coefficients
sigma2 = [sigma(1)*ones(1, ntrain), sigma(2)*ones(1, ntrain*d)].^2;
K = se_kernel_grad(X, params) + diag(sigma2);
lambda = K\vec([Y, DY]);

% Function handle returning GP mean to be output
mu = @(XX) mean_grad(XX, X, lambda, params);
end

function ypred = mean_grad(XX, X, lambda, params)
KK = se_kernel_grad(X, params, XX);
ypred = KK*lambda;

end