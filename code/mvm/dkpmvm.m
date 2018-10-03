% derivative of kronecker product-vector multiplication (A{1} \kron A{2} \kron ... \kron A{n})*x
% Input
%     A: cell of matrices for kronecker product
%     x: target vector for multiplication
%     dA: cell of derivatives
% Output
%     Y: product d/dhyp (A{1} \kron A{2} \kron ... \kron A{n})*x

function dY = dkpmvm(A, x, dA)
    N = length(A);
    dY = zeros(length(x),1);
    for j = 1:N
        dAdi = A;   dAdi(j) = dA(j);
        dY = dY + kpmvm(dAdi, x);
    end
end