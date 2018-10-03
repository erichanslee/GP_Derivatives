% Performs fast TPS kernel mvm on a grid using FFTS in 2 dimensions
% Input
%     xx: x grid points
%     yy: y grid points
%     R: fixed hyperparameter for TPS kernel a positive definite kernel matrix on fixed domain
%     u: target vector for multiplacation
% Output
%     out: vector that is K*u where K is the TPS kernel matrix on a grid defined by xx and yy

function out = bttb2(xx, yy, R, u)
xx = xx - xx(1);
yy = yy - yy(1);
nx = length(xx);
ny = length(yy);

% Reshape xx, yy, and zz to be row vectors
xx = reshape(xx, [1, nx]);
yy = reshape(yy, [1, ny]);


nxp = 2*nx-1; % Number of mesh points including padding
nyp = 2*ny-1;

% Circulant mapping from (-nx,nx) to (1,2*nx-1) index spaces
icx = @(i) 1+mod(i+2*nx-1, 2*nx-1);
icy = @(i) 1+mod(i+2*ny-1, 2*ny-1);

% Write u into padded storage
up = zeros(nxp, nyp);
up(1:nx, 1:ny) = reshape(u,[nx,ny]);

% Set up kernel evaluation matrix
rrx = [-fliplr(xx(2:end)), xx];
rry = [-fliplr(yy(2:end)), yy];

Ix = icx( -(nx-1):(nx-1) )';
Iy = icy( -(ny-1):(ny-1) )';
% s^2*(D.^3 - (3/2)*R*D.^2 + (1/2)*R^3);
kt = reshape(rrx, [nxp, 1, 1]).^2 + reshape(rry, [1, nyp, 1]).^2;
K = kt.^1.5 - (3*R/2)*kt + (1/2)*R^3;
kp(Ix, Iy) = K;

% Do convolution via FFT; throw away padding
yp = real(ifft2(fft2(kp) .* fft2(up)));
y2 = yp(1:nx, 1:ny);

% Sanity check
out = reshape(y2,[nx*ny,1]);
end