function cols = getrow_SKI(mvm, n, k)
    cols = zeros(n, length(k));
    for i = 1:length(k)
        ei = sparse(k(i),1,1,n,1);
        cols(:,i) =  mvm(ei);
    end
end