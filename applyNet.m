function res = applyNet(im, net) 
  im_ = single(im) ; 
  im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
  im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
  %apply network on layer
  res = vl_simplenn(net, im_);
end
