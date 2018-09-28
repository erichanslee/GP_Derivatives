% Script to reconstruct Stanford bunny via implicit surface. **Warning** running this script will take > 30 minutes 
% as training a GP with tens of thousands of data points is computationally expensive. 

clear all
ski_order = 5;
ninduce = 30;
d = 3;

%% Load pre-processed large Stanford bunny (~70,000 vertices) 
% fprintf('Reading... '); obj = readObj('bunny.obj'); fprintf(' Done!\n'); Original Processing Script
load('bunny.mat');


X = obj.v; Xorig = X;
T = obj.f.vt; Torig = T;
nx = obj.vn;

%% Map to unitbox
X = mapToUnitbox(X);
lims = [-0.01 1.01     -0.01 1.01     -0.01 1.01];

%% Normalize the normals
nx = nx ./ sqrt(sum(nx.^2, 2));

%% Add noise
noise = 0.01;
X = X + noise*randn(size(X,1), d);
nx = nx + 0*randn(size(X,1), d);

%% Map back to unitbox and pick subset
X = mapToUnitbox(X);
nn = 5; % use 1/nn of data
x = X(1:nn:end,1); y = X(1:nn:end,2); z = X(1:nn:end,3);
nx = nx(1:nn:end, :);
n = size(x, 1); 
fprintf('Size of Kdot: [%d %d]\n', n*(d+1), n*(d+1))

%% Train GP with gradients using TPS kernel 
lb = min([x,y,z]); ub = max([x,y,z]);
beta = 1e-4; R = 2;
s0 = 1; sig0 = 1e-1;
 
nZ = 3; Z = sign(randn(n*(d+1),nZ));
xg = repmat({linspace(-0.1, 1.1, ninduce)}, d, 1);
[Wtrain{1}, Wtrain{2}] = interpGrid([x,y,z], xg, ski_order);
cov = @(hyp) tps_kernel_grad_ski(R, [x,y,z], hyp, xg, Wtrain);
hyp = struct('cov', log([s0]), 'lik', log([sig0 sig0]));
lmlfun = @(hyp) lml_mvm(cov, [zeros(n, 1), nx], hyp, Z, beta, true);
params = minimize(hyp, lmlfun, -30);
s = exp(params.cov(1));
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('TPS-SKI with gradients: (s, sigma1, sigma2) = (%.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

%% Prediction handle
if length(sigma) == 1, sigma = [sigma, sigma]; end
sig = [sigma(1)*ones(1,n), sigma(2)*ones(1,n*d)]';
[K, ~, precond] = tps_kernel_grad_ski(R, [x,y,z], params, xg, Wtrain);
mvm = @(x) K(x) + sig.^2 .* x;
lambda = pcg(mvm, [zeros(n, 1); nx(:)], 1e-10, 1000, precond);

%% Compute implicit surface
isize = 100;
nxx = isize; nyy = isize; nzz = isize;
x1 = linspace(lims(1), lims(2), nxx); 
x2 = linspace(lims(3), lims(4), nyy); 
x3 = linspace(lims(5), lims(6), nzz);

V = zeros(nxx, nyy, nzz);
for i=1:nzz % Loop over third dimension to not have the memory blow up on us....
    [XX, YY, ZZ] = meshgrid(x1, x2, x3(i));
    Wtest = {}; Wtest{1} = interpGrid([XX(:) YY(:) ZZ(:)], xg, ski_order);
    KK = tps_kernel_grad_ski(R, [x,y,z], params, xg, Wtrain, [XX(:) YY(:) ZZ(:)], Wtest);
    V(:, :, i) = reshape(KK(lambda), [nxx, nyy, 1]); 
end
FV = isosurface(x1, x2, x3, V, 0);

%% Remove vertices far from training points
D = pdist2([x,y,z], FV.vertices,  'euclidean', 'Smallest', 1)';
verticesToRemove = find(D > 3.2e-2)'; 
newVertices = FV.vertices;
newVertices(verticesToRemove,:) = [];
[~, newVertexIndex] = ismember(FV.vertices, newVertices, 'rows');
newFaces = FV.faces(all([FV.faces(:,1) ~= verticesToRemove, ...
    FV.faces(:,2) ~= verticesToRemove, ...
    FV.faces(:,3) ~= verticesToRemove], 2),:);
newFaces = newVertexIndex(newFaces);
V = newVertices;
F = newFaces;

%% Plot original points
figure('units','normalized','outerposition',[0 0 1 1]);
subaxis(1, 3, 1, 'Spacing', 0, 'Padding', 0, 'Margin', 0);
trisurf(Torig, Xorig(:,3), Xorig(:,1), Xorig(:,2), 'EdgeColor', 'none');
axis(lims)
axis equal off
shading interp % Make surface look smooth
view(90, 15)
camlight; lighting phong % Shine light on surface

%% Plot noisy points
subaxis(1, 3, 2, 'Spacing', 0, 'Padding', 0, 'Margin', 0);
trisurf(T, X(:,3), X(:,1), X(:,2), 'EdgeColor', 'none');
axis(lims)
axis equal off
shading interp % Make surface look smooth
view(90, 15)
camlight; lighting phong % Shine light on surface

%% Plot bunny after (hopefully) removing all dummy points
subaxis(1, 3, 3, 'Spacing', 0, 'Padding', 0, 'Margin', 0);
trisurf(F, V(:,3), V(:,1), V(:,2), 'EdgeColor', 'none');
axis(lims)
axis equal off
shading interp % Make surface look smooth
view(90, 15)
camlight; lighting phong % Shine light on surface