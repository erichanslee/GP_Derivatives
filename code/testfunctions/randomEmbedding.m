function [y, dy] = randomEmbedding(x, fcn, Q)
xproj = x*Q;
[y, dy] = fcn(xproj);
dy = dy * Q';
dy = dy .* (1 + 0.05*randn(size(dy)));
end