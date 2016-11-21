function mtx2D = to2D(mtxnD, options)
% convert a h by w by d matrix to d by h*w matrix
[h,w,d] = size(mtxnD);
M_l = h*w;
if isa(mtxnD, 'gpuArray');
  mtx2D = zeros(d, M_l, 'gpuArray');
else
  mtx2D = zeros(d, M_l);
end
for k=1:d
    mtx2D(k,:)=reshape(mtxnD(:,:,k),1,M_l);
end %k
