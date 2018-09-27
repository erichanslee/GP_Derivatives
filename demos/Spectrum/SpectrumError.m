% Script for generating spectral comparisons between D-SKI/D-SKIP and the true SE kernel
clear all
fontsize = 18;
d = 2;
n = 100;
ninduce = 100;

% Training points, sorted for structure in matrix
lb = 0; ub = 1;
xtrain = sortrows(unifrnd(lb, ub, n, d));

% Various Parameters
ell = 0.2;
s = 1;
sigma = 1e-3;
params = struct('cov', log([ell s]), 'lik', log(sigma));
S = [eye(n), zeros(n, n*d); zeros(n*d, n), ell*eye(n*d, n*d)];
figure

%% (1,1) Exact SE with gradients (scaled)
K = se_kernel_grad(xtrain, params) + sigma^2*eye(n*(d+1));
Ktrue = S*K*S;

%% (1,2) D-SKI error
xg = createGrid(xtrain, ninduce);
[Wtrain{1}, Wtrain{2}] = interpGrid(xtrain, xg, 5);
K = se_kernel_grad_ski(xtrain, params, xg, Wtrain, 'exact') + sigma^2*eye(n*(d+1));
K = S*K*S;
subaxis(1, 4, 1, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
imagesc(log10(abs(K - Ktrue)))
hold on;
for i=0:n:(d+1)*n
    for j=0:n:(d+1)*n
        plot(0.5+[0, j], 0.5+[i, i], 'k', 'Linewidth', 2)
        plot(0.5+[i, i], 0.5+[0, j], 'k', 'Linewidth', 2)
    end
end
axis tight off;
colorbar('SouthOutside')
set(gca, 'fontsize', fontsize)
caxis([-10 -3])

%% (2,2) D-SKI spectrum
subaxis(1, 4, 2, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
semilogy(sort(abs(eig(Ktrue)), 'descend'), 'b', 'LineWidth', 3)
hold on
eigVals = sort(abs(eig(K)), 'descend');
inds = [1:17:length(eigVals), length(eigVals)];
semilogy(inds, eigVals(inds), 'r+', 'MarkerSize', 20, 'LineWidth', 3)
legend('True spectrum', 'SKI spectrum')
set(gca, 'fontsize', fontsize)
axis tight

%% (1,3) D-SKIP error
d = 4; n = 200;
xtrain = sortrows(unifrnd(lb, ub, n, d));
ninduce = 150;
xg = cell(1, d); for i=1:d, xg{i} = createGrid(xtrain(:,i), ninduce); end
ell = 0.5;
S = [eye(n), zeros(n, n*d); zeros(n*d, n), ell*eye(n*d, n*d)];

Ktrue = se_kernel_grad(xtrain, params) + sigma^2*eye(n*(d+1));
Ktrue = S*Ktrue*S;

r = 100; % Rank for Lanczos
z = sign(randn(n*(d+1), 1)); % For Hadamard
K = se_kernel_grad_skip(xtrain, params, z, r, xg, 'exact', true) + sigma^2*eye(n*(d+1));
K = S*K*S;

subaxis(1, 4, 3, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
imagesc(log10(abs(K - Ktrue)))
hold on;
for i=0:n:(d+1)*n
    for j=0:n:(d+1)*n
        plot(0.5+[0, j], 0.5+[i, i], 'k', 'Linewidth', 2)
        plot(0.5+[i, i], 0.5+[0, j], 'k', 'Linewidth', 2)
    end
end
axis tight off;
colorbar('SouthOutside')
set(gca, 'fontsize', fontsize)
caxis([-10 -3])

%% (2,3) D-SKIP spectrum
subaxis(1, 4, 4, 'Spacing', 0.01, 'Padding', 0.01, 'Margin', 0.01);
hold off
semilogy(sort(abs(eig(Ktrue)), 'descend'), 'b', 'LineWidth', 3)
hold on
eigVals = sort(abs(eig(K)), 'descend');
inds = [1:70:length(eigVals), length(eigVals)-40, length(eigVals)-7, length(eigVals)]; 
semilogy(inds, eigVals(inds), 'r+', 'MarkerSize', 20, 'LineWidth', 3)
legend('True spectrum', 'SKIP spectrum')
set(gca, 'fontsize', fontsize)
axis tight