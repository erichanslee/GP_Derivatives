% Creates a symmetric Latin hypercube design. 
% 
%Input:
%   d: dimension of the problem
%   m: number of initial points 
%
%Output:
%   InitialPoints: points in the starting design

function InitialPoints = slhd(d,m)

%--------------------------------------------------------------------------

delta = (1/m)*ones(1,d);

X = zeros(m,d);
for j = 1:d
    for i = 1:m
        X(i,j) = ((2*i-1)/2)*delta(j);
    end
end

P = zeros(m,d);
P(:,1) = (1:m)';
if (mod(m,2) == 0)
   k = m/2;
else
   k = (m-1)/2;
   P(k+1,:) = (k+1)*ones(1,d);
end

for j = 2:d
   P(1:k,j) = randperm(k)';
   for i = 1:k
      if (rand(1) <= 0.5)
         P(m+1-i,j) = m+1-P(i,j);
      else
         P(m+1-i,j) = P(i,j);
         P(i,j) = m+1-P(i,j);
      end
   end
end

InitialPoints = zeros(m,d);
for j = 1:d
    for i = 1:m
        InitialPoints(i,j) = X(P(i,j),j);
    end
end

end