clear all;
rng(1111);

numtrials_exact = 10;
numtrials_iter = 10;
d1 = 2;
ssize2 = 500;
hyp.cov(1) = log10(2);
hyp.cov(2) = log10(2);

%% SE grad exact
for i = 1:numtrials_exact
    n = i*ssize2;
    X = rand(n,d1);
    b = rand(n*d1 + n,1);
    
    if( n*(d1+1) < 5*1e4)
        K = se_kernel_grad(X, hyp);
        timings_exact(i) = 0;
        for j = 1:4
            tic;
            mvmb = K*b;
            timings_exact(i) = timings_exact(i) + toc;
        end
        timings_exact(i) = timings_exact(i)/4;
    end
    
end

%% SE SKI grad
for i = 1:numtrials_iter
    n = i*ssize2;
    X = rand(n,d1);
    b = rand(n*d1 + n,1);
    
    timings_se_ski(i) = 0;
    ninduce = floor(n^(1/d1));
    xg = createGrid(X, ninduce);
    [Wtrain{1}, Wtrain{2}] = interpGrid(X, xg, 5);
    mvm =  se_kernel_grad_ski(X, hyp, xg, Wtrain);
    
    for j = 1:4
        tic;
        mvmb_se = mvm(b);
        timings_se_ski(i) = timings_se_ski(i) + toc;
    end
    timings_se_ski(i) = timings_se_ski(i)/4;
    
    
end

d2 = 11;
ssize11 = floor(ssize2*(d1+1)/(d2+1));
hyp.cov(1) = log10(2);
hyp.cov(2) = log10(2);

%% SE SKIP
for i = 1:numtrials_iter
    n = i*ssize11;
    X = rand(n,d2);
    b = rand(n*d2 + n,1);
    
    ninduce = min(round(i*ssize11),100); % Number of inducing points in each dimension
    r = min(round(i*ssize11),100); % Rank for Lanczos
    z = sign(randn(n*(d2+1), 1)); % For Hadamard
    xg = cell(1, d2); for j=1:d2, xg{j} = createGrid(X(:,j), ninduce); end
    mvm = se_kernel_grad_skip(X, hyp, z, r, xg);
    timings_skip(i) = 0;
    
    for j = 1:4
        tic;
        mvmb_skip = mvm(b);
        timings_skip(i) = timings_skip(i) + toc;
    end
    timings_skip(i) = timings_skip(i)/4;
    
end

%% Plot Lines in Loglog10
hold on;
xe1 = log10((d1+1)*ssize2*(1:length(timings_exact)));
plot(xe1, log10(timings_exact), '-b*', 'LineWidth',2);
xe3 = log10((d1+1)*ssize2*(1:length(timings_se_ski)));
plot(xe3, log10(timings_se_ski), '-r*', 'LineWidth',2);
xe4 = log10((d2+1)*ssize11*(1:length(timings_skip)));
plot(xe4, log10(timings_skip), '-m*', 'LineWidth',2);


%% Plot Estimated Complexity Curves

p = polyfit(log10(ssize2*(1:length(timings_exact))), log10(timings_exact), 1);
fprintf('Complexity SE Exact Low D= %.2f\n', p(1))
y = log10(timings_exact(1)*(1:length(timings_exact)).^2) - .1;
plot(xe1, y, '--k','LineWidth',1.5);
text(xe1(end-1),y(end-1),'\leftarrow O(n^2)', 'FontSize',15)

p = polyfit(log10(ssize2*(1:numtrials_iter)), log10(timings_se_ski), 1);
fprintf('Complexity SE D-SKI = %.2f\n', p(1))
y = log10(timings_se_ski(1)*(1:numtrials_iter)) - .85;
plot(xe3, y , '--k','LineWidth',1.5);
text(xe3(end-1),y(end-1),'\leftarrow O(n)','FontSize',15)

p = polyfit(log10(ssize11*(1:numtrials_iter)), log10(timings_skip), 1);
fprintf('Complexity SE D-SKIP= %.2f\n', p(1))
y =  log10(timings_skip(1)*(1:numtrials_iter)) - .05;
plot(xe4, y, '--k','LineWidth',1.5);
text(xe4(end-1),y(end-1),'\leftarrow O(n)','FontSize',15)

set(gca,'fontsize', 15)
l1 = sprintf('SE Exact', d1);
l2 = sprintf('SE SKI (%dD)', d1);
l3 = sprintf('SE SKIP (%dD)', d2);
legend(l1,l2,l3,'Location','northwest');

xlabel('Matrix Size (Log)');
ylabel('MVM Timing (Log)');
title('A Comparison of MVM Scalings');
grid on;

