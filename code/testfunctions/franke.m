function [f, df] = franke(x, lb, ub)
% This function is usually evaluated on [0, 1].
%
% x are points on the unit hypercube
% lb is a d-dimensional vector of bounds of the true domain
% ub is a d-dimensional vector of bounds of the true domain
% Returns f(y) and df(y) where y = lb + (ub-lb) .* x

if nargin == 3
    assert(max(x(:)) <= 1 && min(x(:)) >= 0, 'Points should be in unit hypercube')
    assert(all(lb < ub));
    lb = lb(:)'; ub = ub(:)';
    x = lb + (ub - lb) .* x; % Map points to true domain
end
assert(size(x, 2) == 2, 'This function is in 2D')

term1 = .75*exp(-((9*x(:,1)-2).^2 + (9*x(:,2)-2).^2)/4);
term2 = .75*exp(-((9*x(:,1)+1).^2)/49 - (9*x(:,2)+1)/10);
term3 = .5*exp(-((9*x(:,1)-7).^2 + (9*x(:,2)-3).^2)/4);
term4 = .2*exp(-(9*x(:,1)-4).^2 - (9*x(:,2)-7).^2);
f = term1 + term2 + term3 - term4;
if nargout == 2
    dterm1 = [-2*(9*x(:,1)-2)*9/4.*term1, -2*(9*x(:,2)-2)*9/4.*term1];
    dterm2 = [-2*(9*x(:,1)+1)*9/49.*term2, -9/10.*term2];
    dterm3 = [-2*(9*x(:,1)-7)*9/4.*term3, -2*(9*x(:,2)-3)*9/4.*term3];
    dterm4 = [-2*(9*x(:,1)-4)*9.*term4, -2*(9*x(:,2)-7)*9.*term4];
    df = dterm1 + dterm2 + dterm3 - dterm4;
end

if nargin == 3
    df = df .* (ub - lb);
end
end