% X is a cell with a linspace in each dimension
% Either use {x1, x2, ..., xd} or if the linspaces are the same in each
% dimension you can use {x, d} instead to avoid forming all kernels
%
% Returns the mvm with kernel and hypers

function [mvm, dmvm, precond_handle] = tps_kernel_grid(R, X, hyp)

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