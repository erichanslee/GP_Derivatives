% SKIP with SE kernel and gradients. Note that this kernel is used for training only; 
% with a set of predictive points must be performed using either the exact kernel or a different approximation.
% Input
%     x: training points
%     hyp: hyperparameters
%     z: initial vector for Lanczos
%     r: rank of Lanczos decomposition desired
%     xg: interpolation grid
%     flag: has value 'exact' to get dense matrix instead of mvm if desired (optional)
%     diag_correct: digonal correction of lengthscale < 0.4 (optional)
% Output
%     mvm: mvm with kernel
%     dmvm: mvm with kernel hypers
%     dd: diagonal of kernel
%     get_row: function handle for getting row k

function [mvm, dmvm, dd, get_row] = se_kernel_grad_skip(x, hyp, z, r, xg, flag, diag_correct)


if nargin < 7
    diag_correct = (exp(hyp.cov(1)) < 0.4); % Diagonal correction if ell < 0.4
    if nargin < 6
        flag = [];
    end
end

[n, d] = size(x);
assert(d >= 4, 'Only use SKIP in at least 4D, use SKI in lower dimensions')

s = exp(hyp.cov(2)); % Save s, since otherwise each kernel get their own s
hyp.cov(2) = log(1); % Use s=1 so that there is no contribution from the kernels

if nargin >= 6 && strcmp(flag, 'exact') % Construct dense matrix for testing (this is generally stupid)
    mvm  = s^2*build_mvm(x, hyp, z, r, xg, flag, diag_correct);
    return
end

if nargout == 1
    mvm = build_mvm(x, hyp, z, r, xg, flag, diag_correct);
    mvm = @(x) s^2*mvm(x);
else
    [mvm, dd, get_row, Qtree, Ttree] = build_mvm(x, hyp, z, r, xg, flag, diag_correct);
    dmvm_ell = build_ell_deriv(x, hyp, z, r, xg, Qtree, Ttree, diag_correct);
    mvm = @(x) s^2*mvm(x);
    get_row = @(k) s^2*get_row(k); % Add back s^2
    dd = s^2*dd;
    dmvm = {@(x) s^2 * dmvm_ell(x), @(x) 2*mvm(x)};
end
end

function [mvm, dd, get_row, Qtree, Ttree] = build_mvm(x, hyp, z, r, xg, flag, diag_correct)
tol = 1e-6; % Lanczos truncation tolerance

[n, d] = size(x);
ell = exp(hyp.cov(1));

levels = ceil(log2(d));
Qtree = cell(1, levels);
Ttree = cell(1, levels);

%% Construct 1D SKI approximations (TODO:
W = cell(1, d); dW = cell(1, d); Kg = cell(1, d);
for i=1:d
    [W{i}, dW{i}] = interpGrid(x(:,i), xg{i}, 5);
    [Kg{i}, ~] = se_kernel(xg{i}{1}, hyp);
end

%% Compute active indices on the first level (some magic shuffling, but code has been tested to death)
nn = 2^levels;
act = ones(nn, 1);
inds = flip(2:2:nn); inds = inds(perfect_shuffle(nn/4, 2));
act(inds(1:nn - d)) = 0; inds = find(act);

%% Level 1, compute initial Lanczos factorizations
Q = cell(1, nn); T = cell(1, nn);
for i=1:d
    mvm = @(x) mvm_SKI(Kg{i}, W{i}, dW{i}, n, d, i, x);
    [Q{inds(i)}, T{inds(i)}] = lanczos_arpack(mvm, z, r, tol);
    T{inds(i)} = 0.5*(T{inds(i)} + T{inds(i)}');
    %[Q{i}, T{i}] = truncate_lanczos(Q{i}, T{i});
end
Qtree{1} = Q; Ttree{1} = T;

%% Level 2, where we may be able to skip some factorizations
k = ceil(log2(d)); % Number of levels
nn = nn/2;
for lev=2:k % Recursively build new Lanczos decompositions
    Q = cell(1, nn); T = cell(1, nn);
    for i=1:nn % Set up the mvm
        Q1 = Qtree{lev-1}{2*i-1}; Q2 = Qtree{lev-1}{2*i};
        T1 = Ttree{lev-1}{2*i-1}; T2 = Ttree{lev-1}{2*i};
        if isempty(Q1) % No factorization at position (1, 2i-1)
            Q{i} = Q2;
            T{i} = T2;
        elseif isempty(Q2) % No factorization at position (1, 2i)
            Q{i} = Q1;
            T{i} = T1;
        else % Both factorizations active => Use Lanczos
            Q1T1 = Q1*T1; Q2T2 = Q2*T2;
            mvm1 = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x);
            [Q{i}, T{i}] = lanczos_arpack(mvm1, z, r, tol);
            T{i} = 0.5*(T{i} + T{i}');
            %[Q{i}, T{i}] = truncate_lanczos(Q{i}, T{i});
        end
    end
    Qtree{lev} = Q; Ttree{lev} = T;
    nn = nn/2; % Fewer decompositions next time :)
end

Q1 = Qtree{end}{1}; Q2 = Qtree{end}{2};
T1 = Ttree{end}{1}; T2 = Ttree{end}{2};
%fprintf('r1 = %d, r2 = %d\n', length(T1), length(T2))

% We don't need T explicitly
Q1T1 = Q1*T1;
Q2T2 = Q2*T2;

% We can now set up the final Lanczos mvm!
if nargin >= 6 && strcmp(flag, 'exact')
    mvm = (Q1T1*Q1').*(Q2T2*Q2');
    if diag_correct
        mvm(sub2ind([n*(d+1), n*(d+1)], 1:n*(d+1), 1:n*(d+1))) = [ones(1,n), (1/ell^2)*ones(1,n*d)]'; 
    end
else
    % Diagonal correction (?)
    if diag_correct
        dd_true = [ones(1,n), (1/ell^2)*ones(1,n*d)]';
        dd = sum(Q1T1.*Q1,2) .* sum(Q2T2.*Q2,2); % Current diagonal
        get_row = @(k) getrow2(Q1T1, Q1, Q2T2, Q2, dd_true - dd, k);
        mvm = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x) + (dd_true - dd) .* x;
        dd = dd_true;
    else
        dd = sum(Q1T1.*Q1,2) .* sum(Q2T2.*Q2,2); % Current diagonal
        get_row = getrow(Q1T1, Q1, Q2T2, Q2);
        mvm = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x);
    end
end
end

function rowfun = getrow(Q1T1, Q1, Q2T2, Q2)
rowfun = @(k) (Q1T1(k,:)*Q1').*(Q2T2(k,:)*Q2');
end

function out = getrow2(Q1T1, Q1, Q2T2, Q2, diag_corr, k)
out = (Q1T1(k,:)*Q1').*(Q2T2(k,:)*Q2');
out(k) = out(k) + diag_corr(k);
end

function out = mvm_SKI(Kg, W, dW, n, d, dim, x)
out = zeros(n*(d+1), 1);

%% Batched precomputation of repeated MVMs
WKW = reshape(W*(Kg*(W'*reshape(x, n, d+1))), n, d+1);

%% (1,1) block
out(1:n) = WKW(:,1);

%% (1,2) block
for i=1:d
    if i ~= dim
        out(1:n) = out(1:n) + WKW(:,i+1);
    else
        out(1:n) = out(1:n) + W*(Kg*(dW'*x(1+i*n:(i+1)*n)));
    end
end

%% (2,1) block
for i=1:d
    if i ~= dim
        out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + WKW(:,1);
    else
        out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + dW*(Kg*(W'*x(1:n)));
    end
end

%% (2,2) block
for i=1:d
    for j=1:d
        if i==j && i==dim
            out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + dW*(Kg*(dW'*x(1+j*n:(j+1)*n)));
        elseif i==dim
            out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + dW*(Kg*(W'*x(1+j*n:(j+1)*n)));
        elseif j==dim
            out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + W*(Kg*(dW'*x(1+j*n:(j+1)*n)));
        else
            out(1+i*n:(i+1)*n) = out(1+i*n:(i+1)*n) + WKW(:,1+j);
        end
    end
end
end

function dmvm_ell = build_ell_deriv(x, hyp, z, r, xg, Qtree, Ttree, diag_correct)
tol = 1e-6; % Lanczos truncation tolerance

[n, d] = size(x);
levels = ceil(log2(d));
Q1T1ell = cell(1, d); Q2T2ell = cell(1, d);
Q1ell = cell(1, d); Q2ell = cell(1, d);

dcur = 1;
for i=1:length(Qtree{1}) % Compute ith derivative using product rule
    if isempty(Qtree{1}{i}), continue; end % Skip if factorization doesn't exist
    
    % Compute SKI derivative wrt length scale
    [W, dW] = interpGrid(x(:,dcur), xg{dcur}, 5);
    [~, dKg] = se_kernel(xg{dcur}{1}, hyp);
    dKg = dKg{1};
    
    % Compute top-level Lanczos
    mvm = @(x) mvm_SKI(dKg, W, dW, n, d, dcur, x);
    [Q1, T1] = lanczos_arpack(mvm, z, r, tol);
    T1 = 0.5*(T1 + T1');
    
    % Recurse the tree
    ind = i; % Index of node at current level
    for j=2:levels
        if mod(ind, 2) == 0 % Even, neighboor is to the left
            if ~isempty(Qtree{j-1}{ind-1})
                Q1T1 = Q1*T1; Q2T2 = Qtree{j-1}{ind-1}*Ttree{j-1}{ind-1};
                mvm1 = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Qtree{j-1}{ind-1}, x);
                [Q1, T1] = lanczos_arpack(mvm1, z, r, tol);
                T1 = 0.5*(T1 + T1');
            end
        else % Odd, neighboor is to the right
            if ~isempty(Qtree{j-1}{ind+1})
                Q1T1 = Q1*T1; Q2T2 = Qtree{j-1}{ind+1}*Ttree{j-1}{ind+1};
                mvm1 = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Qtree{j-1}{ind+1}, x);
                [Q1, T1] = lanczos_arpack(mvm1, z, r, tol);
                T1 = 0.5*(T1 + T1);
            end
        end
        ind = ceil(ind/2);
    end
    
    Q1T1ell{dcur} = Q1*T1;
    Q1ell{dcur} = Q1;
    ind = 1 + (2 - ind);
    Q2 = Qtree{end}{ind}; 
    T2 = Ttree{end}{ind};
    Q2T2ell{dcur} = Q2*T2;
    Q2ell{dcur} = Q2;
    
    dcur = dcur + 1; % Move to next dimension
end

if diag_correct
    ell = exp(hyp.cov(1));
    dd = zeros(n*(d+1), d);
    for i=1:d
        dd_true = zeros(n*(d+1), 1); dd_true(1+i*n:(i+1)*n) = -2/ell^2;
        dd(:,i) = dd_true - sum(Q1T1ell{i} .* Q1ell{i}, 2) .* sum(Q2T2ell{i} .* Q2ell{i}, 2); % Current diagonal
    end
    dmvm_ell = @(x) lanczos_sum_mvm2(Q1T1ell, Q1ell, Q2T2ell, Q2ell, dd, x);
else
    dmvm_ell = @(x) lanczos_sum_mvm(Q1T1ell, Q1ell, Q2T2ell, Q2ell, x);
end
end

function y = lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x)
% Compute y = ((Q1*T1*Q1').*(Q2*T2*Q2'))*x in O(n r^2) mvm flops
y = sum( (Q1T1*(Q1'*(x.*Q2T2))) .* Q2, 2); % O(n r1 r2) mvm
end

function y = lanczos_sum_mvm(Q1T1, Q1, Q2T2, Q2, x)
y = zeros(size(x));
for i=1:length(Q1T1)
    y = y + lanczos_mvm(Q1T1{i}, Q1{i}, Q2T2{i}, Q2{i}, x);
end
end

function y = lanczos_sum_mvm2(Q1T1, Q1, Q2T2, Q2, dd, x)

y = zeros(size(x));
for i=1:length(Q1T1)
    y = y + lanczos_mvm(Q1T1{i}, Q1{i}, Q2T2{i}, Q2{i}, x) + dd(:,i) .* x;
end
end