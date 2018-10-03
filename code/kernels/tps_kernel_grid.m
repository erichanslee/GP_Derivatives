%% Thin plate spline (TPS) kernel calculated on a grid
%
% Input
% R: Fixed hyperparameter for TPS kernel, selected to guarantee positive definiteness of kernel on some domain
% X: training points
% hyp: hyperparameters

% 
% Output
% mvm: kernel matrix mvm function handle using d-dimensional FFTs
% dmvm: derivatives of kernel w.r.t. hyperparameters for training

function [mvm, dmvm] = tps_kernel_grid(R, X, hyp)

assert(iscell(X)); d = length(X);
if d > 3
    error('Grid must be of dimension 3 or less');
end

m = length(X{1});
s = exp(hyp.cov(1));
switch d
    case 3
        mvm = @(u) s^2*bttb3(X{1}, X{2}, X{3}, R, u);
        dmvm = {@(u) 2*s^2*bttb3(X{1}, X{2}, X{3}, R, u)};
        n = length(X{1})*length(X{2})*length(X{3});
    case 2
        mvm = @(u) s^2*bttb2(X{1}, X{2}, R, u);
        dmvm = {@(u) 2*s^2*bttb2(X{1}, X{2}, R, u)};
        n = length(X{1})*length(X{2});
end
end