%   [L, P] = piv_chol(A, tol, rr) returns an upper triangular matrix R and a
%   permutation matrix P such that L*L' = P'*A*P. 
%
%   Arguments
%       A: PSD Matrix A of size n x n
%       tol: a lower bound on diagonal elements of L
%       rr: maximum truncation rank rr
%
%   Outputs:
%       L,P: Matrices fufulling L*L' \approx P'*A*P, with equality if 
%            L is a square n by n matrix
%
%   Author: EHL, Cornell University

function [L, P] = pchol(A, tol, rr)

n = length(A);
if nargin == 1, tol = 1e-3, rr = sqrt(n); end

pp = 1:n;
d = diag(A);
L = zeros(n,rr);

for k = 1:rr
    
    % Pick out maximal element from diagonal to pivot
    [dmax, m] = max(d);
    md = m;
    m = m+k-1;
    
    if dmax < 0
        error('Matrix is not PSD, factorization terminated');
    end
    
    
    %   Pivot rows of R
    if m ~= k
        L([k m], :) = L([m k],:);
        pp( [k m] ) = pp( [m k] );
    end
    
    %   Pull out new column, Update Diagonal
    rowk = getrow(A, L(:,1:k), pp, k);
    d([1 md]) = d([md 1]);
    dkk = d(1); 
    d = d(2:end);
    d = d - (rowk.^2)/dkk;
    
    %   Stop factorization if diagonal is less than tol
    %   (i.e. tol is the lower bound on diagonal elements in R)
    if abs(dkk < tol)
        break;
    end

    % Insert new column of into L
    L(k,k) = sqrt( dkk );    
    if k == n, break, end
    L(k+1:n, k) = rowk/L(k,k);
    
end

% Return permutation
P = speye(n); P = P(:,pp);

end


% Given the square A of size n and its mvm, getrow pulls out A(k,k+1:end) via mvm
% L will be an upper triangular matrix of size k by n
function Arow = getrow(A, L, pp, k)

% Using Permutation, extract proper column of A
Ak = A(:,pp(k));
Ak = Ak(pp);
Arow = Ak - L*(L(k,:)');
Arow = Arow(k+1:end);

end

