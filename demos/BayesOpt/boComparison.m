    clear all
    numevals = 100; % We used 500 evals in the paper, which takes quite a bit longer to run
    ntrials = 5;
    dlow = 5;

    %% Test Projected rastragin
    xbfgs = cell(1, ntrials); ybfgs = cell(1, ntrials);
    xrandsamp = cell(1, ntrials); yrandsamp = cell(1, ntrials);
    xbo = cell(1, ntrials); ybo = cell(1, ntrials);
    xbo_ski = cell(1, ntrials); ybo_ski = cell(1, ntrials);


    for i=1:ntrials

        % Get projected function handle
        d = 50; lb = -4*ones(1,d); ub = 5*ones(1,d); f = @(x) rastrigin(x); optimum = -10*dlow;
        [Qproj, ~] = qr(randn(d, dlow), 0);
        f = @(x) randomEmbedding(x, f, Qproj);

        % BFGS
        [xbfgs{i}, ybfgs{i}] = bfgs(f, lb, ub, numevals);
        ybfgs{i} = ybfgs{i}(1:numevals);
        clc

        % Random sampling
        [xrandsamp{i}, yrandsamp{i}] = randsamp(f, lb, ub, numevals);

        % BO exact
        [xbo{i}, ybo{i}] = bo_exact(f, lb, ub, numevals);

        % BO SKI
        [xbo_ski{i}, ybo_ski{i}] = bo_ski(f, lb, ub, numevals);
    end

    %% Plot comparison
    subplot(1,2,1);
    plot(mean(cummin(cell2mat(ybo)), 2), 'LineWidth', 3)
    hold on
    plot(mean(cummin(cell2mat(ybo_ski)), 2), 'r:', 'LineWidth', 3)
    plot(mean(cummin(cell2mat(ybfgs)), 2), 'g-.', 'LineWidth', 3)
    plot(mean(cummin(cell2mat(yrandsamp)), 2), 'm--', 'LineWidth', 3)
    legend('BO exact', 'BO D-SKI', 'BFGS', 'Random sampling')
    title('Embedded 5D Rastragin')
    set(gca, 'fontsize', 30);


%% Test projected Ackley
xbfgs = cell(1, ntrials); ybfgs = cell(1, ntrials);
xrandsamp = cell(1, ntrials); yrandsamp = cell(1, ntrials);
xbo = cell(1, ntrials); ybo = cell(1, ntrials);
xbo_ski = cell(1, ntrials); ybo_ski = cell(1, ntrials);

for i=1:ntrials    
    
    % Get projected function handle
    d = 30; lb = -10*ones(1,d); ub = 15*ones(1,d); f = @ackley;
    [Qproj, ~] = qr(randn(d, dlow), 0);
    f = @(x) randomEmbedding(x, f, Qproj);
    
    % BFGS
    [xbfgs{i}, ybfgs{i}] = bfgs(f, lb, ub, numevals);
    ybfgs{i} = ybfgs{i}(1:numevals);
    clc
    
    % Random sampling
    [xrandsamp{i}, yrandsamp{i}] = randsamp(f, lb, ub, numevals);
    
    % BO exact
    [xbo{i}, ybo{i}] = bo_exact(f, lb, ub, numevals);
    
    % BO SKI
    [xbo_ski{i}, ybo_ski{i}] = bo_ski(f, lb, ub, numevals);
end

%% Plot comparison
subplot(1,2,2);
plot(mean(cummin(cell2mat(ybo)), 2), 'LineWidth', 3)
hold on
plot(mean(cummin(cell2mat(ybo_ski)), 2), 'r:', 'LineWidth', 3)
plot(mean(cummin(cell2mat(ybfgs)), 2), 'g-.', 'LineWidth', 3)
plot(mean(cummin(cell2mat(yrandsamp)), 2), 'm--', 'LineWidth', 3)
legend('BO exact', 'BO D-SKI', 'BFGS', 'Random sampling')
title('Embedded 5D Ackley')
set(gca, 'fontsize', 30);