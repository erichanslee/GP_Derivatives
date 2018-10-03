% Standard thin plate spline (TPS) kernel
% Input
%     R: Fixed hyperparameter for TPS kernel, selected to guarantee positive definiteness of kernel on some domain
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix
%     dKhyp: kernel matrix derivatives w.r.t. hyperparameters

function [K, dKhyp] = tps_kernel(R, X, hyp, XX)
if nargin == 3
    XX = X;
end

s = exp(hyp.cov(1));

D = pdist2(XX, X);
K = s^2*(D.^3 - (3/2)*R*D.^2 + (1/2)*R^3);

if nargout == 2
    dKhyp = {2*K};
end
end