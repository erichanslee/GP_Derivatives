clc
clear all;
rng(1);

%% Settings
tolcg = 1e-4;
maxcgiters = 100;
rankchol = 50;
tolchol = 1e-6;

%% SKI
d = 2; fcn = @franke; lb = [0 0]; ub = [1 1];

% Kernel and hypers
ellvec = logspace(-1, sqrt(d), 15);
sigmavec = logspace(-2, 0, 15);

% Test points
n = 200;
X = rand(n, d);
[y, dy] = fcn(X, lb, ub);

% Center data
y = y - mean(y);

% Create grid
ninduce = 100;
xg = createGrid(X, ninduce);
[Wtrain{1}, Wtrain{2}] = interpGrid(X, xg, 5);

% Test hypers
iters_ski = zeros(length(ellvec), length(sigmavec));
iters_ski_pchol = zeros(length(ellvec), length(sigmavec));
iters_ski_scaled_pchol = zeros(length(ellvec), length(sigmavec));

S = @(hyp) [ones(1,n), exp(hyp.cov(1))*ones(1,n*d)]';
for ii=1:length(ellvec) % Rows
    for jj=1:length(sigmavec) % Columns
        fprintf('(%d, %d)\n', ii, jj)
        
        % Extract hypers
        ell = ellvec(ii); 
        sigma = sigmavec(jj);
        hyp.cov(1) = log(ell);
        hyp.cov(2) = log(1);
        hyp.lik = log(sigma);
        Sigma2 = sigma^2 * ones(n*(d+1), 1);

        % Set up SKI 
        [K, ~, dd, rowhandle] =  se_kernel_grad_ski(X, hyp, xg, Wtrain);
        Ks = @(x) K(x) + sigma^2*x;
        
        % Unscaled system no preconditioner
        [x, fl, rr, it, rv] = pcg(Ks, vec([y, dy]), tolcg, maxcgiters);
        if fl ~= 0, iters_ski(ii,jj) = nan; else, iters_ski(ii,jj) = length(rv); end
        if fl ~= 0, fprintf('CG without a preconditioner did not converge (rr = %.3e)\n', rr); end
            
        % Unscaled system Cholesky preconditioner
        P = pchol_precond(rowhandle, dd, Sigma2, rankchol, tolchol);
        [x, fl, rr, it, rv] = pcg(Ks, vec([y, dy]), tolcg, maxcgiters, P);
        if fl ~= 0, iters_ski_pchol(ii,jj) = nan; else, iters_ski_pchol(ii,jj) = length(rv);  end       
        if fl ~= 0, fprintf('CG with pchol did not converge (rr = %.3e)\n', rr); end
            
        % Scaled system Cholesky preconditioner
        SS = S(hyp);
        scaled_row_handle = @(k) (SS(k) * rowhandle(k) .* SS)'; % Scale row handle
        [L, pp] = pchol_handles(scaled_row_handle, SS .* dd .* SS, tolchol, rankchol);
        P = pchol_solve(pp, L, SS .* Sigma2 .* SS);
        Ks = @(x) SS .* K(SS .* x) + SS .* Sigma2 .* (SS .* x); % Scaled system MVM
        [x, fl, rr, it, rv] = pcg(Ks, SS .* vec([y, dy]), tolcg, maxcgiters, P);
        if fl ~= 0, iters_ski_scaled_pchol(ii,jj) = nan; else, iters_ski_scaled_pchol(ii,jj) = length(rv);  end 
        if fl ~= 0
            fprintf('CG with scaling and pchol did not converge (rr = %.3e) \n', rr); 
        end
    end
end

%% Plot SKI
figure

colormap(hot)
% (1,1) SKI no preconditioner
subaxis(2, 3, 1, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_ski))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKI (No precond)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_ski));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor','r','MarkerSize',8);

% (1,2) SKI with pivoted Cholesky
subaxis(2, 3, 2, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_ski_pchol))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKI (pchol)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_ski_pchol));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor','r','MarkerSize',8);

% (1,3) SKI with pivoted Cholesky and scaling
subaxis(2, 3, 3, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_ski_scaled_pchol))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKI (Scaling + pchol)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_ski_scaled_pchol));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor', 'r', 'MarkerSize', 8);

%% SKIP
% Objective
d = 6; fcn = @hart6; lb = zeros(1, d); ub = ones(1, d);

% Kernel and hypers
ellvec = logspace(-1, sqrt(d), 15);
sigmavec = logspace(-2, 0, 15);

% Test points
n = 200;
X = rand(n, d);
[y, dy] = fcn(X, lb, ub);

% Center data
y = y - mean(y);

% SKIP settings
ninduce = 100; % Number of inducing points in each dimension
r = 50; %round(100*log2(d)); % Rank for Lanczos
z = sign(randn(n*(d+1), 1)); % For Hadamard
xg = cell(1, d); for i=1:d, xg{i} = createGrid(X(:,i), ninduce); end

% Test hypers
iters_skip = zeros(length(ellvec), length(sigmavec));
iters_skip_pchol = zeros(length(ellvec), length(sigmavec));
iters_skip_scaled_pchol = zeros(length(ellvec), length(sigmavec));

S = @(hyp) [ones(1,n), exp(hyp.cov(1))*ones(1,n*d)]';
for ii=1:length(ellvec) % Rows
    for jj=1:length(sigmavec) % Columns
        fprintf('(%d, %d)\n', ii, jj)
        
        % Extract hypers
        ell = ellvec(ii); 
        sigma = sigmavec(jj);
        hyp.cov(1) = log(ell);
        hyp.cov(2) = log(1);
        hyp.lik = log(sigma);
        Sigma2 = sigma^2 * ones(n*(d+1), 1);

        % Set up SKIP
        [K, ~, dd, rowhandle] =  se_kernel_grad_skip(X, hyp, z, r, xg);
        Ks = @(x) K(x) + sigma^2*x;
        
        % Unscaled system no preconditioner
        [x, fl, rr, it, rv] = pcg(Ks, vec([y, dy]), tolcg, maxcgiters);
        if fl ~= 0, iters_skip(ii,jj) = nan; else, iters_skip(ii,jj) = length(rv); end
        if fl ~= 0, fprintf('CG without a preconditioner did not converge (rr = %.3e)\n', rr); end
            
        % Unscaled system Cholesky preconditioner
        P = pchol_precond(rowhandle, dd, Sigma2, rankchol, tolchol);
        [x, fl, rr, it, rv] = pcg(Ks, vec([y, dy]), tolcg, maxcgiters, P);
        if fl ~= 0, iters_skip_pchol(ii,jj) = nan; else, iters_skip_pchol(ii,jj) = length(rv);  end       
        if fl ~= 0, fprintf('CG with pchol did not converge (rr = %.3e)\n', rr); end
            
        % Scaled system Cholesky preconditioner
        SS = S(hyp);
        scaled_row_handle = @(k) (SS(k) * rowhandle(k)' .* SS); % Scale row handle
        [L, pp] = pchol_handles(scaled_row_handle, SS .* dd .* SS, tolchol, rankchol);
        P = pchol_solve(pp, L, SS .* Sigma2 .* SS);
        Ks = @(x) SS .* K(SS .* x) + SS .* Sigma2 .* (SS .* x); % Scaled system MVM
        [x, fl, rr, it, rv] = pcg(Ks, SS .* vec([y, dy]), tolcg, maxcgiters, P);
        if fl ~= 0, iters_skip_scaled_pchol(ii,jj) = nan; else, iters_skip_scaled_pchol(ii,jj) = length(rv);  end 
        if fl ~= 0
            fprintf('CG with scaling and pchol did not converge (rr = %.3e) \n', rr); 
        end
    end
end

%% Plot Stuff
colormap(hot)

% (2,1) SKIP no preconditioner
subaxis(2, 3, 4, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_skip))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKIP (No precond)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_skip));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor','r','MarkerSize',8);

% (2,2) SKIP with pivoted Cholesky
subaxis(2, 3, 5, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_skip_pchol))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKIP (pchol)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_skip_pchol));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor','r','MarkerSize',8);

% (2,3) SKIP with pivoted Cholesky and scaling
subaxis(2, 3, 6, 'Spacing', 0.01, 'Padding', 0.02, 'Margin', 0.02);
pcolor(log10(sigmavec), log10(ellvec), log10(iters_skip_scaled_pchol))
set(gca, 'fontsize', 20)
ylabel('log10(ell)');
title('SKIP (Scaling + pchol)')
colorbar; caxis([0 log10(maxcgiters)]); hold on;
[I,J] = find(isnan(iters_skip_scaled_pchol));
plot(log10(sigmavec(J)), log10(ellvec(I)), 'or', 'MarkerFaceColor', 'r', 'MarkerSize', 8);