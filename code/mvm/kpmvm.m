% Computes kronecker product-vector multiplication (A{1} \kron A{2} \kron ... \kron A{n})*x
% Input
%     A: cell of matrices for kronecker product
%     x: target vector for multiplication
% Output
%     Y: product (A{1} \kron A{2} \kron ... \kron A{n})*x

function Y = kpmvm(A, x)


N = length(A);
if N == 1
    Y = A{1}*x;
elseif N == 2
    k = size(x, 2);
    n = cellfun(@length, A);
    Y = zeros(prod(n), k);
    for i=1:k
        Y(:,i) = reshape(A{2} * reshape(x(:,i), n(2), n(1)) * A{1}', n(1)*n(2), 1);
    end
elseif N == 3
    assert(size(x, 2) == 1);
    n = cellfun(@length, A);
    x = reshape(x, n(2)*n(3), n(1));
    Y = reshape(kpmvm(A(2:3), x*A{1}'), prod(n), 1); % Make recursive call
else
    %Y = kronmult(A, x);
    Y = kronmvm(flip(A), x);
end
end