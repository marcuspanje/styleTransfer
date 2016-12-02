function mtx = vec2mtx(vec,h,w,d)
% convert a d by h*w matrix to a h by w by d matrix
if isa(vec, 'gpuArray')
  mtx = zeros(h,w,d, 'gpuArray');
else
  mtx = zeros(h,w,d);
end

mtx = reshape(vec,h,w,d);


