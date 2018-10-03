% LML_EXACT  Exact log marginal likelihood and derivatives function
% [val, dval] = lml_mvm(cov, y, hyp, Z, precond)
%
% Input:
%   cov: covariance function
%   y: Right hand side (n x (d+1) if gradient information given)
%   hyp: struct of hyper parameters
%   Z: probe vectors for stochastic trace estimator
%   lambda: Regularization added to sigma^2
%
% Output:
%   val: value of lml
%   dval: derivative of lml

function [val, dval] = lml_exact(cov, y, hyp, lambda)


if nargin < 4, lambda = 1e-6; end
[K, dK] = cov(hyp);
nd = length(K); % nd = n*(d+1) with gradients, just n otherwise

% Separate sigma from the other hypers
if isfield(hyp, 'lik')
    if length(hyp.lik) == 2 % Use one sigma for the values, one for the derivatives
        [n, d] = size(y); d = d - 1; assert(d >= 1);
        sigma1 = exp(hyp.lik(1));
        sigma2 = exp(hyp.lik(2));
        sig = [sigma1*ones(1,n), sigma2*ones(1,n*d)];
        [L, p] = chol(K + diag(sig.^2 + lambda), 'lower');
    else
        sigma = exp(hyp.lik); 
        [L, p] = chol(K + (sigma^2 + lambda)*eye(nd), 'lower');
    end
else
    [L, p] = chol(K + lambda*eye(nd), 'lower');
end

if p~=0 % Check if the Cholesky worked
    error('Cholesky failed :/')
end
alpha = L'\(L\y(:));
term1 = y(:)'*alpha; % This is y'*(K\y)
term2 = 2*sum(log(diag(L))); % This is log(det(K))
val = 0.5*(term1 + term2 + nd*log(2*pi));

if nargout >= 2
    for i=1:length(dK) % Derivative wrt kernel hypers
        dval(i) = 0.5*(trace(L'\(L\dK{i})) - alpha'*dK{i}*alpha);
    end
    
    if isfield(hyp, 'lik') % Only return sigma derivative if optimizing noise
        if length(hyp.lik) == 2
            Kinv = L'\(L\eye(nd));
            %sigeye1 = [sigma1^2*eye(n), zeros(n,n*d); zeros(n*d, n*(d+1))];
            %dval(end+1) = trace(L'\(L\sigeye1)) - alpha'*sigeye1*alpha; % Derivative wrt sigma1
            dval(end+1) = sigma1^2*(trace(Kinv(1:n,1:n)) - alpha(1:n)'*alpha(1:n));
            %sigeye2 = [zeros(n, n*(d+1)); zeros(n*d, n), sigma2^2*eye(n*d)];
            %dval(end+1) = trace(L'\(L\sigeye2)) - alpha'*sigeye2*alpha; % Derivative wrt sigma1
            dval(end+1) = sigma2^2*(trace(Kinv(n+1:end,n+1:end)) - alpha(n+1:end)'*alpha(n+1:end));
        else
            dval(end+1) = sigma^2*(trace(L'\(L\eye(nd))) - alpha'*alpha); % Derivative wrt sigma
        end
    end
end
end