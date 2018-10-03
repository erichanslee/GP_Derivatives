% Calculates W and dW, the interpolation weights and derivatives, for SKI and SKI_grad
% Input
%     x: training points
%     xg: grid points
%     deg: degree of interpolant (default is 5)
% Output
%     W: interpolation weights
%     dW: derivative of interpolation weights
    
function [W, dW] = interpGrid(x, xg, deg)

    if nargin < 3,  deg = 5;    end
    [n,d] = size(x);
    ng = zeros(1,d); for i = 1:d, ng(i) = length(xg{i}); end
    N = prod(ng);
    dC = cell(1,d);

    for i = 1:d
        [Ji,Ci,dCi] = interp(x(:,i),xg{i}, deg);
        Ci(abs(Ci)<1e-12) = 1e-12; dCi(abs(dCi)<1e-12) = 1e-12;
        if i == 1
            J = Ji;  C = Ci; dC(1) = {dCi}; dC(2:end) = {Ci};
        else
            pd = prod(ng(1:i-1));
            cp = reshape(repmat((1:size(J,2)),[deg+1 1]),1,[]);
            J = J(:,cp)+repmat((Ji-1)*pd,[1, size(J,2)]);
            C = C(:,cp).*repmat(Ci, [1,size(C,2)]);
            for j = 1:d
                if j == i
                    dC{j} = dC{j}(:,cp).*repmat(dCi, [1,size(dC{j},2)]);
                else
                    dC{j} = dC{j}(:,cp).*repmat(Ci, [1,size(dC{j},2)]);
                end
            end
        end
    end
    I = repmat((1:n)',[1,size(C,2)]);
    W = sparse(I(:), J(:), C(:), n, N);
    s = length(I(:));
    Id = zeros(d*s,1);
    Jd = repmat(J(:),[d 1]);
    value = zeros(d*s,1);
    for i = 1:d
        Id((i-1)*s+1:i*s) = (i-1)*n+I(:); 
        %Id((i-1)*s+1:i*s) = i+d*(I(:)-1); 
        value((i-1)*s+1:i*s) = dC{i}(:);
    end
    dW = sparse(Id, Jd, value, n*d, N);
end

function [J,C,dC] = interp(x, xg, deg)
    if deg == 3, k = @kcub; dk = @dkcub;
    elseif deg == 5, k = @kqui; dk = @dkqui;
    end
    dx = xg(2)-xg(1);
    J = floor((x-xg(1))/dx) - floor(deg/2) + (1:deg+1);
    xi = (x - xg(J))/dx;
    C = k(xi);
    dC = dk(xi)/dx;
end

function y = kcub(x)
% Keys' Cubic Convolution Interpolation Function
  y = zeros(size(x)); x = abs(x);
  q = x<=1;          % Coefficients:  1.5, -2.5,  0, 1
  y(q) =            (( 1.5 * x(q) - 2.5) .* x(q)    ) .* x(q) + 1;
  q = 1<x & x<=2;    % Coefficients: -0.5,  2.5, -4, 2
  y(q) =            ((-0.5 * x(q) + 2.5) .* x(q) - 4) .* x(q) + 2;
end

function y = dkcub(x)
% Keys' Cubic Convolution Interpolation Function Derivative
  y = sign(x); x = abs(x);
  q = x<=1;          % Coefficients:  1.5, -2.5,  0, 1
  y(q) = y(q) .*  ( 4.5 * x(q) - 5.0) .* x(q);
  q = 1<x & x<=2;    % Coefficients: -0.5,  2.5, -4, 2
  y(q) = y(q) .* ((-1.5 * x(q) + 5.0) .* x(q) - 4.0);
  y(x>2) = 0;
end

  function y = kqui(x)
% Keys' Cubic Convolution Interpolation Function
  y = zeros(size(x)); x = abs(x);
  q = x<=1;          % Coefficients:  -0.84375, 1.96875, 0, -2.125, 0, 1
  y(q) =            ((( -0.84375 * x(q) + 1.96875) .* (x(q).^2)) - 2.125) .* (x(q).^2) + 1;
  q = 1<x & x<=2;    % Coefficients: 0.203125, -1.3125, 2.65625, -0.875, -2.578125, 1.90625
  y(q) =            ((((0.203125 * x(q) - 1.3125) .* x(q) + 2.65625) .* x(q) -...
                     0.875) .* x(q) - 2.578125) .* x(q) + 1.90625;
  q = 2<x & x<=3;    % Coefficients: 0.046875, -0.65625, 3.65625, -10.125, 13.921875, -7.59375
  y(q) =            ((((0.046875 * x(q) - 0.65625) .* x(q) + 3.65625) .* x(q) -...
                     10.125) .* x(q) + 13.921875) .* x(q) - 7.59375;          
  end
  
  function y = dkqui(x)
% Keys' Cubic Convolution Interpolation Function
  y = sign(x);  x = abs(x);
  q = x<=1;          % Coefficients:  -4.21875, 7.875, 0, -4.25, 0
  y(q) =    y(q) .* ((( -4.21875 * x(q) + 7.875) .* (x(q).^2)) - 4.25) .* x(q);
  q = 1<x & x<=2;    % Coefficients: 1.015625, -5.25, 7.96875, -1.75, -2.578125
  y(q) =    y(q) .* ((((1.015625 * x(q) - 5.25) .* x(q) + 7.96875) .* x(q) -...
                     1.75) .* x(q) - 2.578125);
  q = 2<x & x<=3;    % Coefficients: 0.234375, -2.625, 10.96875, -20.25, 13.921875
  y(q) =    y(q) .* ((((0.234375 * x(q) - 2.625) .* x(q) + 10.96875) .* x(q) -...
                     20.25) .* x(q) + 13.921875);
  y(x>3) = 0;
  end