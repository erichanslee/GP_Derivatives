% Low rank SVD preconditioner 
% Input
%     Kdot: mvm function handle
%     n: size of matrix
%     Sigma2: noise parameters
%     k: rank of preconditioner
% Output
%     P: function handle performing solve with preconditioner 

function P = svd_precond(Kdot, n, Sigma2, k)
try
    opt.IsSymmetricDefinite = 1;
    [V, D] = eigs(Kdot, n, k, 'LM', opt);
catch
    [V, D] = eigs(Kdot, n, k, 'largestabs', 'IsSymmetricDefinite', 1);
end

if isdiag(Sigma2) || numel(Sigma2) == max(size(Sigma2)) % Hack for faster solves
    if isdiag(Sigma2), dd = diag(Sigma2); else, dd = Sigma2; end
    VV = V./dd;
    L = chol(inv(D) + V'*(V./dd), 'lower'); % Cholesky of the Schur complement 
    P = @(y) y./dd - VV * (L'\(L\(V'* (y./dd))));
else
    error('Hacky hack hack')
    P = @(x) smw_solve(Sigma2, V, D, V, x);
end
end