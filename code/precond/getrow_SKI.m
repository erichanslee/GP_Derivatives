% getrow_SKI pulls out the k-th row of W*Kg*W' for pivoted Cholesky. 
% via mvm with a unit vector (the most efficient way)
% Input
%     mvmv: handle for SKI mvm
%     n: size of matrix
%     k: desired row
%     deg: degree of interpolation (assumed to be 5)
% Output
%     cols: row of matrix (in column vector form, hence the name)

function cols = getrow_SKI(mvm, n, k)
    cols = zeros(n, length(k));
    for i = 1:length(k)
        ei = sparse(k(i),1,1,n,1);
        cols(:,i) =  mvm(ei);
    end
end