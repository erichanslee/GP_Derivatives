% getdiag_SKI pulls out diag(W*Kg*W') for pivoted Cholesky. Calls
% a c function ski_diag.c which must be mexed. 
% Input
%     hyp: hyperparameters
%     xg: interpolation grid
%     Mx: interpolation weights
%     deg: degree of interpolation (assumed to be 5)
% Output
%     dd: diagonal of SKI matrix

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