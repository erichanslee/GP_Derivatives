% Maps points X to the unit box 

function X = mapToUnitbox(X)
minX = min(X); 
lenX = max(X) - minX;
X = (X - minX)./lenX;
end