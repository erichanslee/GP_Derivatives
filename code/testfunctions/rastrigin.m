function [f, df] = rastrigin(x, lb, ub)
% This function is usually evaluated on [-3, 4]^d.
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

[n, d] = size(x);

f = sum(x.^2 - 10*cos(2*pi*x), 2);
if nargout == 2
    df = zeros(n, d);
    for i=1:d
        df(:,i) = 2*x(:, i) + 20*pi*sin(2*pi*x(:,i));
    end
end

if nargin == 3 
    df = df .* (ub - lb); 
end
end