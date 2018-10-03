% Squared Exponential (SE) kernel with gradients and kernel interpolation
% Input
%     X: training points
%     hyp: hyperparameters
%     xg: points for grid interpolation
%     WX: training points weight matrix for interpolant (optional)
%     XX: testing points (optional)
%     WXX: testing poitns weight matrix for interpolant (optional)
% Output
%     K: dense kernel matrix
%     dKhyp: kernel matrix derivatives w.r.t. hyperparameters
%     dd: diagonal of kernel matrix, used for preconditioners
%     get_row: function handle outputting a desired row of kernel matrix, used for preconditioners


function [K, dKhyp, dd, get_row] = se_kernel_grad_ski(X, hyp, xg, WX, XX, WXX)
    if nargin < 4,  [WX{1}, WX{2}] = interpGrid(X, xg, 5);  end
    MX = [WX{1}; WX{2}];
    if nargin < 5 || strcmp(XX,'exact'), MXX = MX; 
    elseif nargin == 5, MXX = interpGrid(XX, xg, 5);
    else,   MXX = WXX{1};  
    end
    [n,d] = size(X);
    
    if nargin == 5 && strcmp(XX, 'exact')
        xe = apxGrid('expand', xg);
        [Kg, dKg] = se_kernel(xe, hyp);
        nhyp = length(dKg);
        dKhyp = cell(1, nhyp);
        K = MXX*Kg*MX';
        for i = 1:nhyp
            dKhyp{i} = MXX*dKg{i}*MX';
        end
        return;
    end
    
    hyp2 = hyp; hyp.cov(2) = hyp.cov(2)/d;
    A = cell(1,d);  dA = cell(1,d);
    for i = 1:d
        [A{d-i+1}, dKhyp] = se_kernel(xg{i}, hyp);
        dA{d-i+1} = dKhyp{1};
    end
    K = @(y) MXX*kpmvm(A,MX'*y);
    if nargout > 1 % For gradient prediction
        dKhyp = {@(y) MXX*dkpmvm(A,MX'*y,dA), @(y) 2*K(y)};
    end
    
    if nargout >= 3
        dd = getdiag_SKI(hyp2.cov, xg, MX);
        get_row = @(k) getrow_SKI(K, n*(d+1), k);
    end
end