%smw_solve solves (A + U*S*V')x=y using the Sherman-Morrison-Woodbury formula

function x = smw_solve(A, U, S, V, y)
Cinv = inv(S) + V'*(A\U);
x = (A\y) - (A\U) * (Cinv\(V'* (A\y)));
end