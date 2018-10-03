% Run BayesOpt using the SKI Kernel and active subspaces, setting the maximal projection dimension (dmax) to be 2
% and using the expected improvement acqusition function 
% Input
%     f: function handle returning both values and derivatives
%     lb: lower bound array
%     ub: upper bound array
%     numevals: evaluation budget
% Output
%     xbfgs: data points evaluated
%     ybfgs: function values evaluated

function [xtrain, ytrain] = bo_ski(f, lb, ub, numevals)
dmax = 2;
d = length(lb);

%% Warm start training points
ndes = d;
xtrain = slhd(d, ndes);
xtrain = lb + xtrain .* (ub - lb);
[ytrain, dytrain] = feval(f, xtrain);

%% Point to evaluate EI
ncand = 1000;
ell = 0.3;
xi = 0.01;
s = std(ytrain);
sigma = 1e-2 * s^2;

for i=1:numevals - ndes
    
    %% Dimensionality reduction
    [Q, dvar] = dim_red(dytrain);
    
    %% Figure out how many dimensions are active (> 99% variance)
    dtilde = find(cumsum(dvar)/sum(dvar) > 0.99, 1, 'first');
    if dtilde > dmax % Pick dmax directions at random
        dtilde = find(dvar/sum(dvar) > 0.05, 1, 'last');
        inds = randperm(dtilde);
        inds = inds(1:dmax);
        dtilde = dmax;
        Q = Q(:, inds);
    else
        Q = Q(:, 1:dtilde);
    end
    
    %% Remove mean from training data
    ymean = mean(ytrain);
    y = ytrain - ymean;
    
    %% Candidate points
    [~, ind] = min(ytrain); xbest = xtrain(ind, :);
    xcand = (0.2*randn(ncand, dtilde) .* max(ub-lb)) * Q' + xbest; % Perturbations in directions of the active subspace
    xcand = min(ub, max(xcand, lb)); % Project on unit hypercube
    
    %% Create grid
    beta = 1e-4; % Regularization added to sigma^2
    nZ = 3;
    xg = createGrid([xtrain*Q; xcand*Q], 50);
    [Wtrain{1}, Wtrain{2}] = interpGrid(xtrain*Q, xg, 5);
    
    %% Kernel learning
    if mod(i, 5) == 1
        Z = sign(randn(length(ytrain)*(dtilde+1), nZ));
        cov = @(hyp) se_kernel_grad_ski(xtrain*Q, hyp, xg, Wtrain);
        lmlfun = @(x) lml_mvm(cov, ([y, dytrain*Q]), x, Z, beta, true, []);
        hyp = struct('cov', log([ell; s]), 'lik', log(sigma));
        params = minimize_quiet(hyp, lmlfun, -30);
        sigma = sqrt(exp(2*params.lik) + beta);
        ell = exp(params.cov(1)); s = exp(params.cov(2));
        fprintf('(%.3f, %.3f, %.3f)\n', exp(params.cov), sigma)
        if s < sigma, s = 10*sigma; params.cov = log([ell s]); end
    end
    
    %% Prediction
    [K, ~, dd, get_row] = se_kernel_grad_ski(xtrain*Q, params, xg, Wtrain);
    Ks = @(x) K(x) + sigma^2 * x;
    [L, pp] = pchol_handles(get_row, dd, 1e-6, min(400, floor(length(ytrain)*(dtilde+1)/10)));
    P = pchol_solve(pp, L, sigma^2);
    [lambda,~] = pcg(Ks, vec([y, dytrain*Q]), 1e-3, 100, P);
    [Wtest{1}, Wtest{2}] = interpGrid(xcand*Q, xg, 5);
    KK = se_kernel_grad_ski(xtrain*Q, params, xg, Wtrain, xcand*Q, Wtest);
    mu = KK(lambda);
    
    %% Predictive variance calculation
    k0 = se_kernel_grad(0, params, 0);  k0 = k0(1); % Value prediction at 0
    p(pp) = 1:size(L, 1);
    Lp = L(p,:);
    [~,pvar] = pcholK_solver(Lp, sigma, k0);
    KYX = se_kernel_grad(xtrain*Q, params, xcand*Q);
    apxD = pvar(KYX);
    stddev = sqrt(max(0, k0 - apxD));
    
    %% Pick the next point
    ei = expected_improvement(mu, stddev, min(y), s*xi);
    [~, maxInd] = max(ei);
    xNext = xcand(maxInd, :);
    [yNext, dyNext] = feval(f, xNext);
    
    %% Append the next point
    xtrain(end+1, :) = xNext;
    ytrain(end+1) = yNext;
    dytrain(end+1, :) = dyNext;
end
end