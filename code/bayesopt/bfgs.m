% A wrapper for fmincon with simple upper and lower bounds. 
% Input
%     f: function handle returning both values and derivatives
%     lb: lower bound array
%     ub: upper bound array
%     numevals: evaluation budget
% Output
%     xbfgs: data points evaluated
%     ybfgs: function values evaluated

function [xbfgs, ybfgs] = bfgs(f, lb, ub, numevals)
xbfgs = []; ybfgs = []; d = length(lb);

while(length(ybfgs) < numevals)
    x0 = lb + (ub - lb) .* rand(1, d);
    options = optimoptions('fmincon', 'SpecifyObjectiveGradient', true, 'MaxIterations', numevals);
    fmincon(@fwrapper, x0, [], [], [], [], lb, ub, [], options);
end

    function [y, dy] = fwrapper(x)
        [y, dy] = f(x);
        xbfgs = [xbfgs; x];
        ybfgs = [ybfgs; y];
    end
end