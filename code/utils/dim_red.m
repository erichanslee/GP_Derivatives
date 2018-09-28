% Finds a new space rotated after decreasing variance
%   - dfx: Derivative observations
%
%   - Q: Basis for the new space
%   - d: d(i) is the variance along Q(:,i)

function [Q, d] = dim_red(dfx)
C = (dfx'*dfx)/size(dfx, 1);
[Q, D] = svd(C);
d = diag(D);
end