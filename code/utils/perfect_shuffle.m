function p = perfect_shuffle(n, d)
p = reshape(reshape(1:n*d, d, n)', n*d, 1);
end

