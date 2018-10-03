% Solves the linear system of equations (K + Sigma2)x = f
% Scaling is important because the noise parameters on the kernel and its hessian
% have different units, so we perform the (S*K*S + S*Sigma2*S)y = S*f, where y = inv(S)*y
% to make things unitless. This function is designed purely for the SE kernel

function [alpha, numiters] = solve_scaled_system(K, S, Sigma2, f, P, tol, maxiter)

if nargin < 7
    maxiter = 500;
    if nargin < 6
        tol = 1e-6;
    end
end

nd = numel(f);
if ~isa(K, 'function_handle'), K = @(x) K*x; end % Wrap K as a handle
Ks = @(x) S .* K(S .* x) + S.* Sigma2 .* (S.*x); % Scaled system MVM

if isempty(P) || (islogical(P) && ~P)
    [alpha, flag, res, it2, rv2] = pcg(Ks, S .* f, tol, maxiter);
    numiters = length(rv2);
    if flag ~= 0, fprintf('Warning: pcg did not converge, residual = %.3e\n', res); end
else
    if isempty(P) % We want to precondition but have no preconditioner
        P = svd_precond(@(x) S .* K(S .* x), nd, S .* Sigma2 .* S, 100);
    end
    [alpha, flag, res, it2, rv2] = pcg(Ks, S .* f, tol, maxiter, P);
    numiters = length(rv2);
    if flag ~= 0, fprintf('Warning: pcg did not converge, residual = %.3e\n', res); end
end

alpha = S .* alpha; % Scale back!
end