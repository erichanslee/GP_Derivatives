% Standard Squared Exponential (SE) kernel
% Input
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix
%     dKhyp: kernel matrix derivatives w.r.t. hyperparameters

function [K, dKhyp] = se_kernel(X, hyp, XX)
if nargin == 2
    XX = X;
end

ell = exp(hyp.cov(1)); 
s = exp(hyp.cov(2));

D = pdist2(XX, X);
K = s^2*exp(-D.^2/(2*ell^2));

if nargout == 2
    dKhyp = {1/ell^2 * (D.^2 .* K), 2*K};
end
end