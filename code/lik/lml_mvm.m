% LML_MVM  MVM based log marginal likelihood and derivatives function
% [val, dval] = lml_mvm(cov, y, hyp, Z, precond, lambda)
%
% Input:
%   cov: covariance function
%   y: Right hand side (n x (d+1) if gradient information given)
%   hyp: struct of hyper parameters
%   Z: probe vectors for stochastic trace estimator
%   lambda: Regularization added to sigma^2
%   precond: Logical true if you want to precondition
%   S: Matrix that scales the system of equations to:
%       (S*K*S)(inv(S)*x) = S*y
%      This is useful in order to put the blocks on the same scale. S can
%      be a function handle that takes hyp as an input, or a fixed matrix.
%
% Output:
%   val: value of lml
%   dval: derivative of lml

function [val, dval] = lml_mvm(cov, y, hyp, Z, lambda, precond, S)


nd = numel(y);
if nargin < 7
    S = [];
    if nargin < 6
        precond = false;
        if nargin < 5
            lambda = 1e-6;
        end
    end
end

if ~islogical(precond), error('precond flag must be logical'); end

% Set up mvms
if precond
    [mvm, dmvm, dd, get_row] = cov(hyp);
else
    [mvm, dmvm] = cov(hyp);
end
    
% Set up the mvms with sigma
if isfield(hyp, 'lik')
    sigma2 = exp(2*hyp.lik);
    
    if length(hyp.lik) == 2 % Use one sigma for the values, one for the derivatives
        [n, d] = size(y); d = d - 1; assert(d >= 1);
        dmvm{end+1} = @(f) [2*sigma2(1)*f(1:n); zeros(n*d, 1)]; % MVM for sigma1 derivative
        dmvm{end+1} = @(f) [zeros(n, 1); 2*sigma2(2)*f(n+1:end)]; % MVM for sigma2 derivative
        sigma2 = [sigma2(1)*ones(1,n), sigma2(2)*ones(1,n*d)]';
    else
        dmvm{end+1} = @(f) 2*sigma2*f; % MVM with sigma derivative
        sigma2 = sigma2 * ones(nd, 1);
    end
    
    sigma2 = sigma2 + lambda; % Add lambda to sigma^2 and compute the new "sigma"
else
    sigma2 = lambda * ones(nd, 1);
end

if isa(S, 'function_handle'), S = S(hyp); end % Construct scaling matrix
if precond
    if isempty(S) % No scaling
        if ~isempty(dd) && ~isempty(get_row)
            [L, pp] = pchol_handles(get_row, dd, 1e-6, ceil(min(nd/10, 500)));
            P = pchol_solve(pp, L, sigma2);
        else
            P = svd_precond(mvm, nd, sigma2, ceil(min(nd/50, 100)));
        end
        mvms = @(x) mvm(x) + sigma2 .* x; % Add lambda to sigma^2
        [alpha, flag, res] = pcg(mvms, y(:), 1e-6, 1000, P);
        if flag ~= 0, fprintf('Warning: pcg did not converge, residual = %.3e\n', res); end
    else % Scaling
        if ~isempty(dd) && ~isempty(get_row)
            scaled_row_handle = @(k) (S(k) * vec(get_row(k)) .* S);
            [L, pp] = pchol_handles(scaled_row_handle, S .* dd .* S, 1e-6, ceil(min(nd/10, 500)));
            P = pchol_solve(pp, L, S .* sigma2 .* S);
        else
            P = svd_precond(@(x) S .* mvm(S .* x), nd, S .* sigma2 .* S, ceil(min(nd/50, 100)));
        end
        alpha = solve_scaled_system(mvm, S, sigma2, y(:), P, 1e-6, 1000);
    end
else
    if isempty(S) % No scaling
        mvms = @(x) mvm(x) + sigma2 .* x; % Add lambda to sigma^2
        [alpha, flag, res] = pcg(mvms, y(:), 1e-6, 1000);
        if flag ~= 0, fprintf('Warning: pcg did not converge, residual = %.3e\n', res); end
    else % Scaling
        alpha = solve_scaled_system(mvm, S, sigma2, y(:), precond);
    end
end

% Do not scale the system for Lanczos
mvms = @(x) mvm(x) + sigma2 .* x; % Add lambda to sigma^2
term1 = y(:)'*alpha;
if nargout == 2
    [ld, dld] = logdet_lanczos(mvms, nd, Z, dmvm, min(nd/2, 100), 1);
else
    ld = logdet_lanczos(mvms, nd, Z);
end
val = 0.5*(term1 + ld + nd*log(2*pi));
dval = zeros(size(dmvm));
if nargout == 2
    for i=1:length(dmvm) % Derivative wrt kernel hypers
        dval(i) = 0.5*(dld(i) - alpha'*dmvm{i}(alpha));
    end
end
end