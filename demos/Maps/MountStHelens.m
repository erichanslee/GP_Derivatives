%% Data Parsing

% Load lidar data
clear all;
load('MtSH.mat');
p = randperm(nx*ny);
trainSize = ceil(0.5*nx*ny);
testSize = nx*ny - trainSize;
X = [mth_points, mth_verts, mth_grads];
d = 2;

% Separate data into training and test sets, making sure to remove mean
Itest = p(1:testSize);
Itrain = p(testSize+1:end);
ntrain = length(Itrain);
ntest = length(Itest);
xtrain = mth_points(Itrain,:); ytrain = mth_verts(Itrain); dytrain = mth_grads(Itrain,:);
xtest = mth_points(Itest,:); ytest = mth_verts(Itest); dytest = mth_grads(Itest,:);
ymean = mean(ytrain);
ytrain = ytrain - ymean;
ytest = ytest - ymean;

%% Various GP training parameters

% Create interpolation grid
ninduce = 80;
xg = createGrid([xtrain;xtest], ninduce);
[Wtrain{1}, Wtrain{2}] = interpGrid(xtrain, xg, 5);
[Wtest{1}, Wtest{2}] = interpGrid(mth_points, xg, 5);

% Stochastic trace estimation probe vectors
nZ = 3; 
Z = sign(randn(length(xtrain),nZ));

% Starting point, no preconditioning necessary
ell0 = 10; s0 = std(ytrain); sig0 = 1e-2*s0; beta = 1e-3;
precond = false;
err = [];

%% SKI with no gradients
cov = @(hyp) se_kernel_ski(xtrain, hyp, xg, Wtrain);
hyp = struct('cov', log([ell0;s0]), 'lik', log(sig0));
lmlfun = @(x) lml_mvm_map(cov, ytrain, x, Z, beta, precond);
params = minimize(hyp, lmlfun, -30);
s2 = exp(2*params.lik) + beta;
fprintf('SKI with no gradients: (ell, s, sigma) = (%.3f, %.3f, %.3f)\n', exp(params.cov), exp(params.lik));

% Prediction
K = se_kernel_ski(xtrain, params, xg, Wtrain);
Ks = @(x) K(x) + s2*x;
lambda = pcg(Ks, ytrain, 1e-3, 1e6);
KK = se_kernel_ski(xtrain, params, xg, Wtrain, mth_points, Wtest);
pred = KK(lambda);
err(end+1) = norm(ytest-pred(Itest))/norm(ytest);

%% SKI with gradients
Z = sign(randn(length(xtrain)*(d+1),nZ)); y = [ytrain; dytrain(:)];
cov = @(hyp) se_kernel_grad_ski(xtrain, hyp, xg, Wtrain);
S = @(hyp)[ones(ntrain,1); exp(hyp.cov(1))*ones(ntrain*2,1)];
lmlfun = @(x) lml_mvm_map(cov, [ytrain,dytrain], x, Z, beta, precond, S);
hyp = struct('cov', log([ell0, s0]), 'lik', log([sig0, sig0]));
params = minimize(hyp, lmlfun, -30);
fprintf('SKI with gradients: (ell, s, sigma1, sigma2) = (%.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), exp(params.lik));

% Prediction
sigma2 = exp(2*params.lik) + beta;
sig = [sigma2(1)*ones(size(xtrain,1), 1); sigma2(2)*ones(size(xtrain,1)*d, 1)];
K = se_kernel_grad_ski(xtrain, params, xg, Wtrain);
Ks = @(x) K(x) + sig.*x;
lambda = pcg(Ks, y, 1e-3, 1e6);
KK = se_kernel_grad_ski(xtrain, params, xg, Wtrain, mth_points, Wtest);
pred_grad = KK(lambda);
pred_grad = pred_grad(1:nx*ny);
err(end+1) = norm(pred_grad(Itest)-ytest)/norm(ytest);

%% Plot
figure('units','normalized','outerposition',[0 0 1 1]);
ypred = [mth_verts-ymean, pred, pred_grad];
names = {'MtSH', 'SKI no gradient','SKI with gradient'};
bot = min(min(ypred));
top = max(max(ypred));

[ha, pos] = tight_subplot(1,3,[.01 .03],[.1 .1],[.01 .01]);
for i = 1:size(ypred,2)
    axes(ha(i));
    title(names{i})
    set(gca,'fontsize', 20)
    contour(reshape(ypred(:,i),[nx, ny]),60);
    axis off
end

fprintf('SKI with no gradient Prediction Error: %.6f\n', err(1));
fprintf('SKI with gradient Prediction Error: %.6f\n', err(2));