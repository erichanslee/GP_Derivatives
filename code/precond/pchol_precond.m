% Pivoted Cholesky Based-Preconditioner P = pchol_precond(Kdot, sigma, k)
%
% Arguments:
% rowhandle -- function handle extracting row
% dd -- diagonal work
% Sigma2 -- Diagonal Matrix
% k -- desired rank (optional and fixed to 30 otherwise)
% tol --desired tolerance (optional and fixed to 1e-3)
% Outputs:
% Function handle giving a solver for Kdot_k + sigma^2*I using SMW, where 
% Kdot_k is a k-rank approximation of Kdot

function P = pchol_precond(rowhandle, dd, Sigma2, k, tol)

if(nargin == 3)
    k = 10;
    tol = 1e-3;
end

[L, pp] = pchol_handles(rowhandle, dd, tol, k);
P = pchol_solve(pp, L, Sigma2);
end