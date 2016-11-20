function mtx2D = to2D(mtxnD)
% convert a h by w by d matrix to d by h*w matrix
[h,w,d] = size(mtxnD);
mtx2D = [];
M_l = h*w;
for k=1:d
    mtx2D(end+1,:)=reshape(mtxnD(:,:,k),1,M_l);
end %k