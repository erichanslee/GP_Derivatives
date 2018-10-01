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