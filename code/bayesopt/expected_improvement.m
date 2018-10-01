function ei = expected_improvement(mu, stddev, target, xi)
% Compute the expected improvement given prediction, uncertainty, target
%   NB: target is most often picked to be min f(x_i)
%   You can add a xi that increases the applicatios

if nargin < 4, xi=0; end
gamma = ((target+xi) - mu)./stddev;
ei = stddev .* gamma .* normcdf(gamma) + stddev .* normpdf(gamma);
ei(stddev < 1e-6) = 0; % Just to be safe
if ei < 0, fprintf('WTF?!?!?!?!?\n'); end
end