inputFace;
%%% inputFace above %%%

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
imR =  applyNet(im0, net); 

disp('generating new image');
%reference layer
L = 1;
gradNext = imR(L+1).x - imC(L+1).x;
step = 0.01;%gradient des step size
Niterations = 20;
for iter = 1:Niterations
  %calculate error by back-propagation
  gradNext = imR(L+1).x - imC(L+1).x;
  if mod(iter, 2) == 0
    err = gradNext.^2;
    err = sum(sum(sum(err)));
    disp(sprintf('iteration %03d, err: %d', iter, err));
  end

  for layer = fliplr(1:L)
    type = net.layers{layer}.type;
    szYprev = size(imR(layer).x);
    grad = zeros(szYprev);

    if strcmp(type, 'conv')
      weights = net.layers{layer}.weights{1};
      pad = net.layers{layer}.pad;
      stride = net.layers{layer}.stride;
      Yprev = imR(layer).x;
      [grad,~,~] = vl_nnconv(Yprev, weights, [], gradNext, ...
        'pad', pad, 'stride', stride);

    elseif strcmp(type, 'softmax')
      %gradient = softmaxGD(params) * gradient;
    elseif strcmp(type, 'relu')
      %gradient = reluGD(params) * gradient;
    elseif strcmp(type, 'pool')
      %gradient = reluGD(pool) * graident;
    end 
    gradNext = grad;

  end %for each layer

  imR(1).x = imR(1).x - step*grad;
  %reapply network on image
  imR = vl_simplenn(net, imR(1).x);
end % for each iteration

imRDisp= uint8(bsxfun(@plus, imR(1).x, net.meta.normalization.averageImage));
figure(1);
subplot(121);
imshow(im); %original image
title('reference');
subplot(122);
imshow(imRDisp);
title('generated');
