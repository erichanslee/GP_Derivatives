% SKIP is a kernel used for training only. Once training is performed and hyperparameters/interpolation coefficients are calculated, 
% this function allows one to perform prediction via SKIP and D-SKIP
% Input
%     xtrain: training points
%     hyp: hyperparamters
%     xg: grid points
%     xtest: prediction points
%     lambda: interpolation coefficients
% Output
%     ypred: predicted values at xtest

function ypred = predict_skip(xtrain, hyp, xg, xtest, lambda)
ntrain = size(xtrain, 1);
ntest = size(xtest, 1);
d = size(xtrain, 2);
s = exp(hyp.cov(2)); 
hyp.cov(2) = log(1);

% Value predictions
KK = s^2 * ones(ntest, ntrain);
for i=1:d
    Kg = se_kernel(xg{i}{1}, hyp);
    WX = interpGrid(xtrain(:,i), xg{i}, 5);
    WXX = interpGrid(xtest(:,i), xg{i}, 5);
    KK = KK .* (WXX * Kg * WX');
end
ypred = KK*lambda(1:ntrain);

% Gradient predictions
for dim=1:d
    KK = s^2 * ones(ntest, ntrain);
    for i=1:d
        Kg = se_kernel(xg{i}{1}, hyp);
        [WX, dWX] = interpGrid(xtrain(:,i), xg{i}, 5);
        WXX = interpGrid(xtest(:,i), xg{i}, 5);
        if i == dim
            KK = KK .* (WXX * Kg * dWX');
        else
            KK = KK .* (WXX * Kg * WX');
        end
    end
    ypred = ypred + KK * lambda(1+dim*ntrain:(dim+1)*ntrain);
end