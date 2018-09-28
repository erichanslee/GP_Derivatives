% Interface for training a GP using vanilla SKI using the SE Kernel
% 
% [mu, K] = gp_SKI(X, Y)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: mvm handle for SKI kernel K(X,X)

function [mu, K] = gp_SKI(X, Y)

[ntrain, d] = size(X);
if(ntrain < 100)
    error('Not that many training points; we recommend you use the exact kernel instead');
end

% Various parameters and initial hyperparameters
ell0 = 0.2*sqrt(d);
s0 = std(Y); 
beta = 1e-4; 
sig0 = 5e-2*std(Y); 
precond = true; 

% Hard code number of inducing points
if d==1, ninduce=100; elseif d==2, ninduce=100; elseif d==3, ninduce=50; ... 
else error('SKI does not scale well with dimension > 3'); end

% Create interpolation grid
xg = createGrid(X, ninduce);
[Wtrain{1}, Wtrain{2}] = interpGrid(X, xg, 5);

% Create three probe vectors for stochastic trace estimation
nZ = 3; 
Z = sign(randn(ntrain,nZ)); 

% Set kernel and loss function 
cov = @(hyp) se_kernel_ski(X, hyp, xg, Wtrain);
lmlfun = @(x) lml_mvm(cov, Y, x, Z, beta, precond);
hyp = struct('cov', log([ell0 s0]), 'lik', log(sig0));

% Optimize hyperparameters
params = minimize_quiet(hyp, lmlfun, -50);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SKI no gradients: (ell, s, sigma) = (%.3f, %.3f, %.3f)\n', exp(params.cov), sigma);

% Calculate lambda = inv(K + sigma2*I)*f via i.e. the interpolation coefficients
% with conjugate gradient and a pivoted cholesky preconditioner
sigma2 = sigma^2 * ones(ntrain, 1);
[K, ~, dd, get_row] = se_kernel_ski(X, params, xg, Wtrain);
[L, pp] = pchol_handles(get_row, dd, 1e-10, min(400, floor(ntrain/10)));
P = pchol_solve(pp, L, sigma2);
Ks = @(x) K(x) + sigma2 .* x;
[lambda,~] = pcg(Ks, Y, 1e-10, 1e6, P);

% Function handle returning GP mean to be output
mu = @(XX) mean_SKI(XX, X, lambda, params, xg, Wtrain);
end

function ypred = mean_SKI(XX, X, lambda, params, xg, Wtrain)
% Create separate interpolation grid for training points
[Wtest{1}, Wtest{2}] = interpGrid(XX, xg, 5);

% Prediction
KK = se_kernel_ski(X, params, xg, Wtrain, XX, Wtest);
ypred = KK(lambda);
end