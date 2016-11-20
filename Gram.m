function gram = Gram(F)
% F is the matrix of N features of layer L
[r,c] = size(F);
gram = zeros(r);
for i=1:r
    for j=1:r
        gram(i,j) = F(i,:) * F(j,:)';
    end %j
end %i