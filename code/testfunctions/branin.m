function [f, df] = branin(x, lb, ub)
% This function is usually evaluated on [-5, 10] x [0, 15].
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

a = 1; b = 5.1/(4*pi^2); c = 5/pi; r = 6; s = 10; t = 1/(8*pi);
f = a*(x(:,2)-b*x(:,1).^2+c*x(:,1)-r).^2 + s*(1-t)*cos(x(:,1)) + s;
if nargout == 2
    df = [2*a*(-2*b*x(:,1)+c).*(x(:,2)-b*x(:,1).^2+c*x(:,1)-r)-s*(1-t)*sin(x(:,1)), ...
        2*a*(x(:,2)-b*x(:,1).^2+c*x(:,1)-r)];
end

if nargin == 3
    df = df .* (ub - lb);
end

end