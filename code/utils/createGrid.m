% Taken from GPML: http://www.gaussianprocess.org/gpml/code/matlab/doc/
% to greate uniform interpolation grid from training points x

function xg = createGrid(x, k)
  if nargin<2, k = 1; end                                     % set input params
  p = size(x,2); xg = cell(p,1);                               % allocate result
  k = ones(p,1).*k(:);
  for j=1:p                                            % iterate over dimensions
    u = sort(unique(x(:,j))); if numel(u)<2, error('Two few unique points.'),end
    if isempty(k)
        ngj = ceil( (u(end)-u(1))/min(abs(diff(u))) );     % use minimum spacing
    elseif 0<=k(j) && k(j)<=1
      ngj = ceil(k(j)*numel(u));
    else
      ngj = k(j);
    end
    du = (u(end)-u(1))/(ngj); bu = [u(1)-7*du, u(end)+7*du];                                                     % equispaced grid
    xg{j} = linspace(bu(1),bu(2),max(ngj,5))';        % at least 5 grid points
  end
end