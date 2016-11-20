function mtxnD = toND(mtx2D,h,w)
% convert a d by h*w matrix to a h by w by d matrix
[d,~] = size(mtx2D);

mtxnD = zeros(h,w,d);

for k=1:d
    
    mtxnD(:,:,k) = reshape(mtx2D(k,:),h,w);
   
end %k