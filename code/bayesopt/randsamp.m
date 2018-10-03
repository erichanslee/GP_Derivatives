% Optimizer than just randomly chooses points, used as baseline in BO experiment
% Input
%     f: function handle returning both values and derivatives
%     lb: lower bound array
%     ub: upper bound array
%     numevals: evaluation budget
% Output
%     xbfgs: data points evaluated
%     ybfgs: function values evaluated

function [xrand, yrand] = randsamp(f, lb, ub, numevals)
d = length(lb);
xrand = zeros(numevals, d);
yrand = zeros(d, 1);
for i=1:numevals
    xrand(i, :) = lb + (ub - lb) .* rand(1, d);
    yrand(i) = f(xrand(i, :));
end
end