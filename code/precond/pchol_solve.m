% Returns a handle for a solver calculating
% P*L*L^T*P^T + Sigma2*x = b
% Where P is a permutation matrix and L is a cholesky factor

function P = pchol_solve(pp, L, Sigma2)
[~,kk] = size(L);

n = length(pp);
p(pp) = 1:n;
Lp  = L(p,:);

assert(isdiag(Sigma2) || numel(Sigma2) == max(size(Sigma2)))
if isdiag(Sigma2), dd = diag(Sigma2); else, dd = Sigma2; end

%% Fast inaccurate SMW solve
% UU = Lp./dd;
% L = chol(eye(kk) + Lp'*(Lp./dd), 'lower'); % Cholesky of the Schur complement
% P = @(y) y./dd - UU * (L'\(L\(Lp'* (y./dd))));

%% Slower but accurate QR solve
[Q,~] = qr([Lp./sqrt(dd); eye(kk)], 0);
Q1 = Q(1:n,:)./sqrt(dd);
P = @(y) y./dd - Q1*(Q1'*y);
end