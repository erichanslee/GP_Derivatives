function [f, df] = ackley(x, lb, ub)
% This function is usually evaluated on  [-3, ..., -3] x [4, ..., 4]. 
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
f = -20*exp(-0.2*sqrt(sum(x.^2,2)/d)) - exp(sum(cos(2*pi*x),2)/d);

if nargout == 2
    df = zeros(n, d);
    for i=1:d
        df(:,i) = 4*x(:,i).*exp(-0.2*sqrt(sum(x.^2,2)/d))./(d*sqrt(sum(x.^2,2)/d)) + ...
           2*pi*sin(2*pi*x(:,i))/d.*exp(sum(cos(2*pi*x),2)/d);
    end
    df(isnan(df)) = 0; % Only if x = 0
end

if nargin == 3 
    df = df .* (ub - lb); 
end
end