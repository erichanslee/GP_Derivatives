% SKIP with SE kernel. Note that this kernel is used for training only; 
% with a set of predictive points must be performed using either the exact kernel or a different approximation.
% Input
%     x: training points
%     hyp: hyperparameters
%     z: initial vector for Lanczos
%     r: rank of Lanczos decomposition desired
%     xg: interpolation grid
%     flag: has value 'exact' to get dense matrix instead of mvm if desired (optional)
% Output
%     mvm: mvm with kernel
%     dmvm: mvm with kernel hypers
%     dd: diagonal of kernel
%     get_row: function handle for getting row k

function [mvm, dmvm, dd, get_row] = se_kernel_skip(x, hyp, z, r, xg, flag)

% Return a fast MVM to the product kernel with given mvms
[n, d] = size(x); % Number of dimensions
assert(d >= 4, 'Only use SKIP in at least 4D, use SKI in lower dimensions')

s = exp(hyp.cov(2)); % Save s, since otherwise each kernel get their own s
hyp.cov(2) = log(1); % Use s=1 so that there is no contribution from the kernels

if nargin == 6 && strcmp(flag, 'exact')
    mvm = s^2 * build_mvm(x, hyp, z, r, xg, 'exact');
    return;
end

[mvm, dd, get_row] = build_mvm(x, hyp, z, r, xg);
get_row = @(k) s^2*get_row(k); % Add back s^2
dd = s^2*dd;

mvm = @(x) s^2*mvm(x);
if nargout >= 2 % Use finite differences for now :(
    h = 1e-6;
    hyp2 = hyp; hyp2.cov(1) = hyp.cov(1) + h; % Remember we want to differentiate wrt log(ell)
    mvm2 = build_mvm(x, hyp2, z, r, xg);
    dmvm = {@(x) (1/h)*(s^2*mvm2(x) - mvm(x)), @(x)2*mvm(x)};
end

end

function [mvm, dd, get_row] = build_mvm(x, hyp, z, r, xg, flag)
[n, d] = size(x); % Number of dimensions
mvms = cell(1, d); % Set up the mvms
for i=1:d
    [Wtrain{1}, Wtrain{2}] = interpGrid(x(:,i), xg{i}, 3);
    mvm = se_kernel_ski(x(:,i), hyp, xg{i}, Wtrain);
    mvms{i} = mvm;
end


% Level 1, compute initial Lanczos factorizations
Q = cell(1, d);
T = cell(1, d);
for i=1:d
    [Q{i}, T{i}] = lanczos_arpack(@(x) mvms{i}(x), z, r);
    [Q{i}, T{i}] = truncate_lanczos(Q{i}, T{i});
    %[Q{i}, T{i}] = lanczos_fast(@(x) mvms{i}(x), z, r);
end

% Level 2, where we may be able to skip some factorizations
nn = 2^(ceil(log2(d)));
act = ones(nn, 1);
inds = flip(2:2:nn); inds = inds(perfect_shuffle(nn/4, 2));
act(inds(1:nn - d)) = 0;
next = 1;
nn = nn/2;
for i=1:nn % Loop over the level
    if act(2*(i-1)+1) == 1 && act(2*i) == 1
        Q1 = Q{next}; Q1T1 = Q1*T{next};
        Q2 = Q{next+1}; Q2T2 = Q2*T{next+1};
        mvm1 = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x);
        [Q{i}, T{i}] = lanczos_arpack(mvm1, z, r);
        [Q{i}, T{i}] = truncate_lanczos(Q{i}, T{i});
        %[Q{i}, T{i}] = lanczos_fast(mvm1, z, r);
        next = next+2;
    else
        Q{i} = Q{next};
        T{i} = T{next};
        next = next+1;
    end
end

% Loop over remaining levels
k = ceil(log2(d)); % Number of levels
nn = nn/2;
for lev=3:k % Recursively build new Lanczos decompositions
    for i=1:nn % Set up the mvm
        Q1 = Q{2*(i-1)+1}; Q1T1 = Q1*T{2*(i-1)+1};
        Q2 = Q{2*i}; Q2T2 = Q2*T{2*i};
        mvm1 = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x);
        [Q{i}, T{i}] = lanczos_arpack(mvm1, z, r);
        [Q{i}, T{i}] = truncate_lanczos(Q{i}, T{i});
        %[Q{i}, T{i}] = lanczos_fast(mvm1, z, r);
    end
    nn = nn/2; % Fewer decompositions next time :)
end

% Wipe out small elements (!?!?!???!)
[Q1, T1] = truncate_lanczos(Q{1}, T{1});
[Q2, T2] = truncate_lanczos(Q{2}, T{2});
fprintf('r1 = %d, r2 = %d\n', length(T{1}), length(T{2}))

% We don't need T explicitly
Q1T1 = Q1*T1;
Q2T2 = Q2*T2;

% We can now set up the final Lanczos mvm!
if nargin == 6 && strcmp(flag, 'exact')
    mvm = (Q1T1*Q1').*(Q2T2*Q2');
    %mvm(sub2ind([n, n],1:n, 1:n)) = 1;
else
    % Diagonal correction
    dd = sum(Q1T1.*Q1,2) .* sum(Q2T2.*Q2,2); % Current diagonal
    mvm = @(x) lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x); % + (1 - dd) .* x;
    get_row = getrow(Q1T1, Q1, Q2T2, Q2);
end
end

function rowfun = getrow(Q1T1, Q1, Q2T2, Q2)
    rowfun = @(k) (Q1T1(k,:)*Q1').*(Q2T2(k,:)*Q2');
end

function [Q, T] = truncate_lanczos(Q, T)
%ind1 = 1:find(diag(T) > 1e-5 * max(diag(T)), 1, 'last');
ind1 = 1:find(diag(T(2:end,2:end)) < 1e-6 * max(diag(T)), 1, 'first');
if numel(ind1) == 0, ind1 = 1:size(T,1); end
Q = Q(:, ind1);
T = T(ind1, ind1); 
end

function y = lanczos_mvm(Q1T1, Q1, Q2T2, Q2, x)
% Compute y = ((Q1*T1*Q1').*(Q2*T2*Q2'))*x in O(n r^2) mvm flops
y = sum( (Q1T1*(Q1'*(x.*Q2T2))) .* Q2, 2); % O(n r1 r2) mvm
end