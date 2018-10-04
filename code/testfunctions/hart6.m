function [f, df] = hart6(X, lb, ub)
% This function is usually evaluated on [0, 1]^6.
%
% x are points on the unit hypercube
% lb is a d-dimensional vector of bounds of the true domain
% ub is a d-dimensional vector of bounds of the true domain
% Returns f(y) and df(y) where y = lb + (ub-lb) .* x

if nargin == 3
    assert(max(X(:)) <= 1 && min(X(:)) >= 0, 'Points should be in unit hypercube')
    assert(all(lb < ub));
    lb = lb(:)'; ub = ub(:)';
    X = lb + (ub - lb) .* X; % Map points to true domain
end
assert(size(X, 2) == 6, 'This function is in 6D')

[n, d] = size(X);

% Define alpha, A, and P
alpha = [1.0, 1.2, 3.0, 3.2]';
A = [10, 3, 17, 3.5, 1.7, 8;
    0.05, 10, 17, 0.1, 8, 14;
    3, 3.5, 1.7, 10, 17, 8;
    17, 8, 0.05, 10, 0.1, 14];
P = 10^(-4) * [1312, 1696, 5569, 124, 8283, 5886;
    2329, 4135, 8307, 3736, 1004, 9991;
    2348, 1451, 3522, 2883, 3047, 6650;
    4047, 8828, 8732, 5743, 1091, 381];

% Compute f
f = zeros(n, 1);
for j = 1:4
    aj = A(j,:);
    pj = P(j,:);
    temp = (X - pj).^2;
    f = f - alpha(j)*exp(-temp*aj');
end

% Compute df
if nargout == 2
    df = zeros(n, d);
    for j = 1:4
        aj = A(j,:);
        pj = P(j,:);
        temp = (X - pj).^2;
        for k = 1:6
            df(:, k) = df(:, k) + 2*alpha(j)*A(j,k)*(X(:,k) - P(j,k)).*exp(-temp*aj');
        end
    end
end

if nargin == 3
    df = df .* (ub - lb); % Scale derivatives back to unit hypercube
end
end