% getdiag_SKI pulls out diag(W*Kg*W')
% Mx can be W or [W;dW]
function dd = getdiag_SKI(hyp, xg, Mx, deg)
    if nargin < 4, deg = 5; end % assume quintic interpolation
    KgVec = [];
    d = length(xg);
    ng = zeros(d,1);
    ell = exp(hyp(1));
    s = exp(hyp(2)/d);
    for i = d:-1:1
        ng(i) = length(xg{i});
        colone = s^2*exp(-(xg{i}-xg{i}(1)).^2/(2*ell^2));
        KgVec = [KgVec; colone];
    end
    [rows,~,vals] = find(Mx');
    dd = ski_diag(vals, int32(rows), KgVec, int32(ng), int32(deg));
end