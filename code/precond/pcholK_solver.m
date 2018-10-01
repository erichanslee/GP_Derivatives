% [solver, pvar] = pcholK_solver(L, sigma, phi0)
%
% Given a partial Cholesky factor L, compute a solver and a predictive
% variance function for Khat = (sigma^2 I + L L').
%
% Inputs:
%   L:     n-by-k partial Cholesky factor
%   sigma: noise parameter
%   phi0:  kernel diagonal element
%
% Output
%   solver(b): function to compute Khat\b
%   pvar(KUX): function to compute diag(KUX*Khat\KUX')
%
function [solver, pvar] = pcholK_solver(L, sigma, phi0)

  [n,k] = size(L);
  if prod(size(sigma)) == 1
  
    % No diagonal correction / constant sigma
    [Q,R] = qr([L; sigma*eye(k)], 0);
    Q1 = Q(1:n,:);
    d = (1-sum(Q1.^2,2))/sigma^2;
    solver = @(b) (b-Q1*(Q1'*b))/sigma^2;
    pvar   = @(KUX) phi0-(sum(KUX.^2,2)-sum((KUX*Q1).^2,2))/sigma^2;
    
  else

    % Include diagonal correction or varying sigma
    Ds = spdiags(1./sigma,0,n,n);
    Di = spdiags(1./(sigma.^2), 0, n,n);
    [Q,R] = qr([Ds*L; eye(k)], 0);
    Q1 = Ds*Q(1:n,:);
    solver = @(b) Di*b-Q1*(Q1'*b);
    pvar   = @(KUX) phi0-((KUX.^2)*(1./(sigma.^2))-sum((KUX*Q1).^2,2));

  end