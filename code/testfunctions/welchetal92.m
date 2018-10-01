function [f, df] = welchetal92(x, lb, ub)

% lb = -0.5*ones(1,20);
% ub = 0.5*ones(1,20);

if nargin < 3, error('Sorry guys, need to be careful with the domain scaling'); end
assert(max(x(:)) <= 1 && min(x(:)) >= 0, 'Points should be in unit hypercube')
assert(all(lb < ub));
assert(size(x, 2) == 20, 'This function is in 20D')

lb = lb(:)'; ub = ub(:)';
x = lb + (ub - lb) .* x; % Map points to true domain
f = welchetal92_inner(x);

df = zeros(size(x));
h = 1e-6;
for i = 1:20
    ei = zeros(size(x));
    ei(:,i) = h;
    fd2 = welchetal92_inner(x + ei);
    fd1 = welchetal92_inner(x - ei);
    df(:,i) =  (fd2 - fd1)/(2*h);
end
df = df .* (ub - lb);

end

function f = welchetal92_inner(xx)
x1  = xx(:,1);
x2  = xx(:,2);
x3  = xx(:,3);
x4  = xx(:,4);
x5  = xx(:,5);
x6  = xx(:,6);
x7  = xx(:,7);
x8  = xx(:,8);
x9  = xx(:,9);
x10 = xx(:,10);
x11 = xx(:,11);
x12 = xx(:,12);
x13 = xx(:,13);
x14 = xx(:,14);
x15 = xx(:,15);
x16 = xx(:,16);
x17 = xx(:,17);
x18 = xx(:,18);
x19 = xx(:,19);
x20 = xx(:,20);

term1 = 5.*x12 ./ (1+x1);
term2 = 5 .* (x4-x20).^2;
term3 = x5 + 40.*x19.^3 - 5.*x19;
term4 = 0.05.*x2 + 0.08.*x3 - 0.03.*x6;
term5 = 0.03.*x7 - 0.09.*x9 - 0.01.*x10;
term6 = -0.07.*x11 + 0.25.*x13.^2 - 0.04.*x14;
term7 = 0.06.*x15 - 0.01.*x17 - 0.03.*x18;

f = term1 + term2 + term3 + term4 + term5 + term6 + term7;
end