% Computes kronecker product-vector multiplication (A{1} \kron A{2} \kron ... \kron A{n})*x
% Input
%     As: cell of matrices for kronecker product
%     x: target vector for multiplication
%     transp: flag to perform mvm with tranposed matrix (optional)
% Output
%     Y: product (A{1} \kron A{2} \kron ... \kron A{n})*xfunction b = kronmvm(As,x,transp)

function b = kronmvm(As,x,transp)
if nargin>2 && ~isempty(transp) && transp   % transposition by transposing parts
  for i=1:numel(As)
    if isnumeric(As{i})
      As{i} = As{i}';
    else
      As{i}.mvm = As{i}.mvmt;
      As{i}.size = [As{i}.size(2),As{i}.size(1)];
    end
  end
end
m = zeros(numel(As),1); n = zeros(numel(As),1);                  % extract sizes
for i=1:numel(n)
  if isnumeric(As{i})
    [m(i),n(i)] = size(As{i});
  else
    m(i) = As{i}.size(1); n(i) = As{i}.size(2);
  end
end
d = size(x,2);
b = x;
for i=1:numel(n)                              % apply As{i} to the 2nd dimension
  sa = [prod(m(1:i-1)), n(i), prod(n(i+1:end))*d];                        % size
  a = reshape(permute(reshape(full(b),sa),[2,1,3]),n(i),[]);
  if isnumeric(As{i}), b = As{i}*a; else b = As{i}.mvm(a); end    % do batch MVM
  b = permute(reshape(b,m(i),sa(1),sa(3)),[2,1,3]);
end
b = reshape(b,prod(m),d);                        % bring result in correct shape