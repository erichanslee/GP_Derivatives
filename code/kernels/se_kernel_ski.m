%% SKI kernel no gradients implementation
%
% Input
% X: training points
% hyp: hyperparameters
% xg: grid points for interpolation
% WX: interpolation weights training
% XX: testing points (optional, needed for prediction)
% WXX: interpolation weights testing (optional, needed for prediction)
% 
% Output
% K: kernel matrix mvm function handle
% dKhyp: derivatives of kernel w.r.t. hyperparameters for training
% dd: diagonal of matrix, necessary for pivoted cholesky preconditioner
% get_row: function handle outputting row of ski, necessary for pivoted cholesky preconditioner



function [K, dKhyp, dd, get_row] = se_kernel_ski(X, hyp , xg, WX, XX, WXX)
    if nargin < 4,  WX{1} = interpGrid(X, xg, 5);  end
    if nargin < 5 || strcmp(XX,'exact'), WXX = WX; 
    elseif nargin == 5, WXX{1} = interpGrid(XX, xg, 5); end
    [n,d] = size(X);
    
    if nargin == 5 && strcmp(XX, 'exact')
        xe = apxGrid('expand', xg);
        [Kg, dKg] = se_kernel(xe, hyp);
        nhyp = length(dKg);
        dKhyp = cell(1, nhyp);
        K = WXX{1}*Kg*WX{1}';
        for i = 1:nhyp
            dKhyp{i} = WXX{1}*dKg{i}*WX{1}';
        end
        return;
    end
    
    hyp2 = hyp; hyp.cov(2) = hyp.cov(2)/d;
    A = cell(1,d);  dA = cell(1,d);
    for i = 1:d
        [A{d-i+1}, dKhyp] = se_kernel(xg{i}, hyp);
        dA{d-i+1} = dKhyp{1};
    end
    K = @(y) WXX{1}*kpmvm(A,WX{1}'*y);
    if nargout > 1 % For gradient prediction
        dKhyp = {@(y) WX{1}*dkpmvm(A,WX{1}'*y,dA), @(y) 2*K(y)};
    end
    
    if nargout >= 3
        dd = getdiag_SKI(hyp2.cov, xg, WX{1});
        get_row = @(k) getrow_SKI(K, n, k);
    end
end