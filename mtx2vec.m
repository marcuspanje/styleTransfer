function vec = mtx2vec(mtx, options)
% convert a h by w by d matrix to d by h*w matrix
[h,w,d] = size(mtx);
len = h*w*d;
if isa(mtx, 'gpuArray');
  vec = zeros(len,1, 'gpuArray');
else
  vec = zeros(len,1);
end

vec = reshape(mtx,len,1);
