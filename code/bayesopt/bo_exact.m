% A standard bayesian optimization routine fitting using SE kernel with gradients and 
% picking points with the expected improvement acquisition function
% Input
%     f: function handle returning both values and derivatives
%     lb: lower bound array
%     ub: upper bound array
%     numevals: evaluation budget
% Output
%     xtrain: data points evaluated
%     ytrain: function values evaluated

function [xtrain, ytrain] = bo_exact(f, lb, ub, numevals)
d = length(lb);

%% Training points
ndes = d;
xtrain = slhd(d, ndes);
xtrain = lb + xtrain .* (ub - lb);
[ytrain, dytrain] = feval(f, xtrain);

%% Point to evaluate EI
ncand = 10000;
xi = 0.01; % Forces exploration

ell = 0.3*norm(ub - lb);
s = std(ytrain);
sigma = 1e-3 * s^2;
ei_max = nan*ones(ndes, 1);

for i=1:numevals - ndes
    %% Remove mean from training data
    ymean = mean(ytrain);
    y = ytrain - ymean;
    ny = length(y);
    
    %% Fit exact GP with gradients
    if mod(i, 5) == 1
        reg = 1e-6; % Regularization added to sigma^2
        hyp = struct('cov', log([ell; s]), 'lik', log(sigma));
        cov = @(hyp) se_kernel(xtrain, hyp);
        lmlfun = @(x) lml_exact(cov, y, x, reg);
        params = minimize_quiet(hyp, lmlfun, -50);
        sigma = sqrt(exp(2*params.lik) + reg);
        ell = exp(params.cov(1)); s = exp(params.cov(2));
        if s < sigma, s = 10*sigma; params.cov = log([ell s]); end
    end
    Sigma2 = sparse(1:ny, 1:ny, sigma.^2);
    
    %% Candidate points
    xcand = lb + (ub - lb) .* rand(ncand, d);
    
    %% Prediction
    K = se_kernel(xtrain, params) + Sigma2;
    lambda = K\y;
    KK = se_kernel(xtrain, params, xcand);
    mu = KK*lambda;
    k0 = se_kernel(0, params, 0);  k0 = k0(1); % Value prediction at 0
    KYX = se_kernel(xtrain, params, xcand)';
    stddev = sqrt(max(0, k0 - sum(KYX' .* (K\KYX)', 2))); % Uncertainty
    
    %% Expected improvement (remember that mean is removed)
    ei = expected_improvement(mu, stddev, min(y), s*xi);
    fprintf('(%d) %.3f %.3e\n', i, min(ytrain), max(ei))
    
    %% Pick the next point
    [ei_max(end+1), maxInd] = max(ei);
    xNext = xcand(maxInd, :);
    [yNext, dyNext] = feval(f, xNext);
    
    %% Append the next point
    xtrain(end+1, :) = xNext;
    ytrain(end+1) = yNext;
    dytrain(end+1, :) = dyNext;
end
end