% [ldB, dldB] = logdet_lanczos(B, n, nZ, dB, kmax, orth)
%
% Approximate the logdet and its derivative of a kernel matrix through Lanczos
% iteration
%
% Input:
%   B: Matrix multiplication
%   n: Dimension of the space
%   nZ: Number of probe vectors with which we want to compute moments
%   dB: Derivative. Cell array of function handles
%   kmax: Number of Lanczos steps
%   orth: No reorthogonalization or partial reorthogonaliztion
%
% Output:
%   ldB: log(det(B)) estimation
%   dldB: derivative with respect to hyperparameters
%
function [ldB,dldB] = logdet_lanczos(B, n, nZ, dB, kmax, orth)
    if nargin < 6,  orth = 1;  end
    if nargin < 5,  kmax = 100;  end
    if nargin < 3,  nZ = ceil(log(n)); end
        
    if length(nZ) > 1
        Z = nZ;
        nZ = size(Z,2);
    else
        Z = sign(randn(n,nZ));
    end
    
    if nargout > 1
        dBZ = zeros(n, nZ*length(dB));
        for i = 1:length(dB)
            for j = 1:nZ
                dBZ(:,(i-1)*nZ+j) = dB{i}(Z(:,j)); 
            end
        end
        dldB = zeros(nZ,length(dB));
    end
    
    ldB = zeros(nZ,1);
    
    if orth
        OpLanz = @(z) lanczos_arpack(B, z, kmax);
    else
        OpLanz = @(z) lanczos_fast(B, z, kmax);
    end
    
    for k = 1:nZ
        [Q,T] = OpLanz(Z(:,k));
        nT = length(T);
        [V,theta] = eig(T,'vector');
        wts = (V(1,:).').^2 * norm(Z(:,k))^2;
        ldB(k) = sum(wts.*log(theta));
        if nargout > 1
            Kinvz = Q*(T\[1;zeros(nT-1,1)]);
            dldB(k,:) = norm(Z(:,k))*sum(bsxfun(@times,Kinvz,dBZ(:,k:nZ:end)));
        end
    end
    ldB = real(mean(ldB, 1));
    if nargout > 1, dldB = real(mean(dldB, 1)); end
end
