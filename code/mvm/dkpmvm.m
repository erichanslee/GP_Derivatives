% derivative of kronecker product
% A = {A1, A2,..., AN}
% dA(i) = {dAi/dhyp1,...,dAi/dhypm}
%
function dY = dkpmvm(A, x, dA)
    N = length(A);
    dY = zeros(length(x),1);
    for j = 1:N
        dAdi = A;   dAdi(j) = dA(j);
        dY = dY + kpmvm(dAdi, x);
    end
end