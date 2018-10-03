%   [L, P] =pchol_handles(rowhandle, dd, tol, rr) returns an upper triangular matrix R and a
%   permutation matrix P such that L*L' = P'*A*P.
%
%   Arguments
%       rowhandle: handle providing row of matrix
%       dd: diagonal of matrix
%       tol: a lower bound on diagonal elements of L
%       rr: maximum truncation rank rr
%
%   Outputs:
%       L,P: Matrices fufulling L*L' \approx P'*A*P, with equality if
%            L is a square n by n matrix
%

function [L, pp] = pchol_handles(rowhandle, dd, tol, rr)

% Get dimensions
n = length(dd);

% Pull Diagonal from full matrix implicitly

if nargin == 2, tol = 1e-3; rr = sqrt(n); end

pp = 1:n;
L = zeros(n,rr);

for k = 1:rr
    
    % Pick out maximal element from diagonal to pivot
    [dmax, m] = max(dd);
    md = m;
    m = m+k-1;
    
    if dmax < 0
        break;
    end
    
    
    %   Pivot rows of R
    if m ~= k
        L([k m], :) = L([m k],:);
        pp( [k m] ) = pp( [m k] );
    end
    
    %   Pull out new column, Update Diagonal
    Ak = rowhandle(pp(k));
    if size(Ak, 1) > 1 && size(Ak, 2) > 1, error('Row handle outputing matrix(?!?!?!?)'); end
    if size(Ak, 1) == 1 && size(Ak, 2) > 1, Ak = Ak'; end % Transpose if row
    Ak = Ak(pp);
    Arow = Ak - L*(L(k,:)');
    rowk = Arow(k+1:end);
    dd([1 md]) = dd([md 1]);
    dkk = dd(1);
    dd = dd(2:end);
    dd = dd - (rowk.^2)/dkk;
    
    %   Stop factorization if diagonal is less than tol
    %   (i.e. tol is the lower bound on diagonal elements in R)

    if abs(sum(dd(2:end))) < tol
        break;
    end
    
    % Insert new column of into L
    L(k,k) = sqrt( dkk );
    if k == n, break, end
    L(k+1:n, k) = rowk/L(k,k);
    
end
end