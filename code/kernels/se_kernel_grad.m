% Standard Squared Exponential (SE) kernel with gradients
% Input
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix
%     dKhyp: kernel matrix derivatives w.r.t. hyperparameters


function [K, dKhyp] = se_kernel_grad(X, hyp, XX)
assert(~(nargout == 2 && nargin == 3), 'Prediction + two outputs not compatible')

pred = true;
if nargin == 2
    XX = X;
    pred = false;
end

ell = exp(hyp.cov(1)); 
s = exp(hyp.cov(2));

[n, d] = size(X);
[n2, ~] = size(XX);

%% (1) Kernel block
if pred, K=zeros(n2, n*(d+1)); else, K = zeros(n*(d+1), n*(d+1)); end
D = pdist2(XX, X);
K(1:n2, 1:n) = exp(-D.^2/(2*ell^2));

if nargout == 2
    dKhyp = zeros(n*(d+1), n*(d+1));
    dKhyp(1:n, 1:n) = 1/ell^2 * (D.^2 .* K(1:n, 1:n));
end

%% (2) Gradient block
for i=1:d
    K(1:n2, 1+i*n:(i+1)*n) = (XX(:,i) - X(:,i)')/ell^2.*K(1:n2, 1:n);
    
    if nargout == 2
        dKhyp(1:n, 1+i*n:(i+1)*n) = (X(:,i) - X(:,i)')/ell^2.*dKhyp(1:n, 1:n) ...
            -2*(X(:,i) - X(:,i)')/ell^2.*K(1:n, 1:n);
    end
end

if pred
    K = s^2*K;
    return
end

%% (3) Gradient block again
K(1+n:end, 1:n) = K(1:n, 1+n:end)';
if nargout == 2
    dKhyp(1+n:end, 1:n) = dKhyp(1:n, 1+n:end)';
end

%% (4) Hessian block
for i=1:d
    for j=1:d
        K(1+i*n:(i+1)*n, 1+j*n:(j+1)*n) = ...
            ((i==j)/ell^2 - (X(:,i)-X(:,i)').*(X(:,j)-X(:,j)')/ell^4).*K(1:n, 1:n);
        
        if nargout == 2
            dKhyp(1+i*n:(i+1)*n, 1+j*n:(j+1)*n) = ...
                ((i==j)/ell^2 - (X(:,i)-X(:,i)').*(X(:,j)-X(:,j)')/ell^4).*dKhyp(1:n, 1:n) + ...
                (-2*(i==j)/ell^2 + 4*(X(:,i)-X(:,i)').*(X(:,j)-X(:,j)')/ell^4).*K(1:n, 1:n);
        end
    end
end

K = s^2*K;
if nargout == 2
    dKhyp = {s^2*dKhyp, 2*K};
end
end