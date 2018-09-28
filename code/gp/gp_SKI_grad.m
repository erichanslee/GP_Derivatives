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

function [mu, K] = gp_SKI_grad(X, Y, DY)

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

% Create interpolation grid and weights
xg = createGrid(X, ninduce);
[Wtrain{1}, Wtrain{2}] = interpGrid(X, xg, 5);

% Create three probe vectors for stochastic trace estimation
nZ = 3; 
Z = sign(randn(ntrain*(d+1),nZ)); 

% Set kernel and loss function 
cov = @(hyp) se_kernel_grad_ski(X, hyp, xg, Wtrain);
S = @(hyp) [ones(1,ntrain), exp(hyp.cov(1))*ones(1,ntrain*d)]'; % Diagonal scaling
lmlfun = @(x) lml_mvm(cov, ([Y, DY]), x, Z, beta, precond, S);
hyp = struct('cov', log([ell0 s0]), 'lik', log([sig0, sig0]));

% Optimize hyperparameters
params = minimize_quiet(hyp, lmlfun, -50);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE-SKI with gradients: (ell, s, sigma1, sigma2) = (%.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), sigma);

% Calculate lambda = inv(K + sigma2*I)*f via i.e. the interpolation coefficients
% with conjugate gradient and a pivoted cholesky preconditioner
S = S(params);
sigma2 = [sigma(1)*ones(1, ntrain), sigma(2)*ones(1, ntrain*d)]'.^2;
[K, ~, dd, get_row] = se_kernel_grad_ski(X, params, xg, Wtrain);
scaled_row_handle = @(k) (S(k)*vec(get_row(k)) .* S); % Scale row handle
[L, pp] = pchol_handles(scaled_row_handle, S .* dd .* S, 1e-10, min(400, floor(ntrain*(d+1)/10)));
P = pchol_solve(pp, L, S .* sigma2 .* S);
lambda = solve_scaled_system(K, S, sigma2, vec([Y, DY]), P, 1e-10);

% Function handle returning GP mean to be output
mu = @(XX) mean_SKI_grad(XX, X, lambda, params, xg, Wtrain);
end

function ypred = mean_SKI_grad(XX, X, lambda, params, xg, Wtrain)

% Create separate interpolation grid for training points
[Wtest{1}, Wtest{2}] = interpGrid(XX, xg, 5);

% Prediction
KK = se_kernel_grad_ski(X, params, xg, Wtrain, XX, Wtest);
ypred = KK(lambda);

end