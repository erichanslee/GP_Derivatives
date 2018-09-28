% Interface for training a GP using vanilla SKI with gradients using the SE Kernel
% 
% [mu, K] = gp_SKI_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: mvm handle for SKI kernel K(X,X)

function [mu, K] = gp_SKIP_grad(X, Y, DY)


[ntrain, d] = size(X);
if(ntrain < 100)
    error('Not that many training points; we recommend you use the exact kernel instead');
end

% Starting Points
ell0 = 0.2*sqrt(d); 
s0 = std(Y); 
sig0 = 5e-2*s0;
beta = 1e-3;
precond = true;
ninduce = 100; % Number of inducing points in each dimension
r = 100; % Rank for Lanczos when training

% Initialize interpolation grid 
z = sign(randn(ntrain*(d+1), 1)); % For Hadamard
xg = cell(1, d); for i=1:d, xg{i} = createGrid(X(:,i), ninduce); end
S = @(hyp) [ones(1,ntrain), exp(hyp.cov(1))*ones(1,ntrain*d)]';

% Stochastic trace estimation variables
nZ = 3; Z = sign(randn(ntrain*(d+1), nZ));

% Set kernel and loss function 
cov = @(hyp) se_kernel_grad_skip(X, hyp, z, r, xg);
lmlfun = @(x) lml_mvm(cov, ([Y, DY]), x, Z, beta, precond, S);
hyp = struct('cov', log([ell0 s0]), 'lik', log([sig0, sig0]));

% Optimize hyperparameters
params = minimize_quiet(hyp, lmlfun, -30);
sigma = sqrt(exp(2*params.lik) + beta);
if length(sigma) == 1, sigma = [sigma, sigma]; end
fprintf('SKIP with gradients: (ell, s, sigma1, sigma2) = (%.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Solve for lambda
S = S(params);
r = ceil(100*sqrt(d)); % Rank for Lanczos when calculating interpolation coefficients (higher than training for greater precision)
Sigma2 = [sigma(1)*ones(1, ntrain), sigma(2)*ones(1, ntrain*d)]'.^2;
[K, ~, dd, get_row] = se_kernel_grad_skip(X, params, z, r, xg);
scaled_row_handle = @(k) (S(k)*get_row(k)' .* S)'; 
rank_precond =  min(1000, floor(ntrain*(d+1)/10));
[L, pp] = pchol_handles(scaled_row_handle, S .* dd .* S, 1e-10, rank_precond);
P = pchol_solve(pp, L, S .* Sigma2 .* S);
lambda = solve_scaled_system(K, S, Sigma2, vec([Y, DY]), P, 1e-10, 1000);

% Calculate lambda = inv(K + sigma2*I)*f via i.e. the interpolation coefficients
% with conjugate gradient and a pivoted cholesky preconditioner
mu = @(XX) mean_SKIP_grad(XX, X, lambda, params);

end

function ypred = mean_SKIP_grad(XX, X, lambda, params)

% Prediction, note that we can predict with either a different approximate kernel or the true one
% after having solved for lambda. We opt for the true kernel.
KK = se_kernel_grad(X, params, XX); % Exact prediction is cheaper
ypred = KK*lambda;

end