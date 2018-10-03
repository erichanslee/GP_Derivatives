% Run lanczos without reorthogonalization, which is faster but less stable
% Input
%     Afun: mvm function handle
%     z: random initial vector
%     kmax: rank of Lanczos
% Output
%     Q, T such that A = Q'*A*Q

function [Q,T] = lanczos_fast(Afun, z, kmax)
    if nargin < 3, kmax = 150; end
    % initialization
    n = length(z);
    Q = zeros(n, kmax);
    alpha = zeros(kmax, 1);
    beta = zeros(kmax, 1);

    k  = 0;
    qk = zeros(n, 1);
    n1 = norm(z);
    r  = z/n1;
    b  = 1;
  
    % Lanczos algorithm without reorthogonalization
    while k < kmax
        k = k+1;
        qkm1 = qk;
        qk = r/b;
        Q(:,k) = qk;
        Aqk = Afun(qk);
        alpha(k) = qk'*Aqk;
        r = Aqk - alpha(k)*qk - b*qkm1;
        b = norm(r);
        beta(k) = b;
    end
    T = diag(alpha) + diag(beta(1:end-1),1) + diag(beta(1:end-1),-1);
end