% Compute the expected improvement given prediction, uncertainty, target
% Input
%     mu: mean array
%     stddev: variance array
%     target: current minimum found
%     xi: perturbation to target to encourage exploration, default value is zero
% Output
%     ei: array of expected improvements
    
function ei = expected_improvement(mu, stddev, target, xi)

if nargin < 4, xi=0; end
gamma = ((target+xi) - mu)./stddev;
ei = stddev .* gamma .* normcdf(gamma) + stddev .* normpdf(gamma);
ei(stddev < 1e-6) = 0; % truncate small expected improvements
end