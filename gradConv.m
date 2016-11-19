function grad = gradConv(weights, pad, stride, Yprev, gradNext)
  szW = size(weights);
  szYprev = size(Yprev);
  [grad,~,~] = vl_nnconv(Yprev, weights, [], gradNext, ...
    'pad', pad, 'stride', stride);
end
