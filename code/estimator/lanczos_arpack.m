% Arpack Lanczos wrapper
% Input
%     B: mvm function handle
%     v: random initial vector
%     d: rank of Lanczos
%     eps: tolerance
% Output
%     Q, T such that A = Q'*A*Q
    
function [Q,T] = lanczos_arpack(B, v, d, eps)     % perform Lanczos with at most d MVMs
  if nargin<4, eps = 1e-10; end
  n = length(v);
  v = bsxfun(@times,v,1./sqrt(sum(v.*v,1)));               % avoid call to normc
  ido = uint64(0); nev = uint64(ceil((d+1)/2)); ncv = uint64(d+1);
  ldv = uint64(n); info = uint64(1);
  lworkl = uint64(ncv*(ncv+8));
  iparam = zeros(11,1,'int64'); ipntr = zeros(15,1,'int64');
  arpackc_reset();
  iparam([1,3,7]) = [1,300,1]; tol = 1e-10;
  Q = zeros(n,ncv); workd = zeros(n,3); workl = zeros(lworkl,1); count = 0;
  amax = -inf; flag = true;
  while ido~=99 && count<=d && flag
    count = count+1;
    [ido,info] = arpackc('dsaupd',ido,'I',uint64(n),'LM',nev,tol,v,...
                          ncv,Q,ldv,iparam,ipntr,workd,workl,lworkl,info);
    if abs(workl(ncv+count-1)) < eps*amax, flag = false; end
    amax = max(amax, abs(workl(ncv+count-1)));
    if info<0
      error(message('ARPACKroutineError',aupdfun,full(double(info))));
    end
    if ido == 1, inds = double(ipntr(1:3));
    else         inds = double(ipntr(1:2)); end
    rows = mod(inds-1,n)+1; cols = (inds-rows)/n+1; % referenced column of ipntr
    if ~all(rows==1), error(message('ipntrMismatchWorkdColumn',n)); end
    switch ido                       % reverse communication interface of ARPACK
      case -1, workd(:,cols(2)) = B(workd(:,cols(1)));
      case  1, workd(:,cols(3)) =   workd(:,cols(1));
               workd(:,cols(2)) = B(workd(:,cols(1)));
      case  2, workd(:,cols(2)) =   workd(:,cols(1));
      case 99                                                        % converged
      otherwise, error(message('UnknownIdo'));
    end
  end
%   ncv = int32(ncv);
  Q = Q(:,1:count-1);                                            % extract results
  T = diag(workl(ncv+1:ncv+count-1))+diag(workl(2:count-1),1)+diag(workl(2:count-1),-1);