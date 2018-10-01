function [xrand, yrand] = randsamp(f, lb, ub, numevals)
d = length(lb);
xrand = zeros(numevals, d);
yrand = zeros(d, 1);

for i=1:numevals
    xrand(i, :) = lb + (ub - lb) .* rand(1, d);
    yrand(i) = f(xrand(i, :));
end
end