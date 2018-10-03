%% Thin plate spline (TPS) kernel with gradients and kernel interpolation
%
% Input
% R: Fixed hyperparameter for TPS kernel, selected to guarantee positive definiteness of kernel on some domain
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

function [K, dKhyp, dd, get_row] = tps_kernel_grad_ski(R, X, hyp, xg, WX, XX, WXX)
    dd = []; get_row = [];
    
    if nargin < 5,  [WX{1}, WX{2}] = interpGrid(X, xg, 5);  end
    MX = [WX{1}; WX{2}];
    if nargin < 6 || strcmp(XX,'exact'), MXX = MX; 
    elseif nargin == 6, MXX = interpGrid(XX, xg, 5);
    else,   MXX = WXX{1};  
    end
    [n,d] = size(X);
    
    if nargin == 6 && strcmp(XX, 'exact')
        xe = apxGrid('expand', xg);
        [Kg, ~] = tps_kernel(R, xe, hyp);
        K = MXX*Kg*MX';
        dKhyp = {2*K};
        return;
    end
    
    f = tps_kernel_grid(R, xg, hyp);
    K = @(y) MXX*f(MX'*y);
    dKhyp = {@(x) 2*K(x)};
end