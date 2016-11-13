%replace path with location of vl_setupnn from matlabconvnet package
run('~/code/matconvnet-1.0-beta21/matlab/vl_setupnn');
%load weights of the trained vgg-face network
%this repo does not store the mat file. It can be obtianed from:
%http://www.robots.ox.ac.uk/~vgg/data/vgg_face/
net = load('vgg-face.mat');
net = vl_simplenn_tidy(net);

%images must be 244x244
im = imread('img/khan.jpg');
im_ = single(im) ; 
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
%apply network on layer
res = vl_simplenn(net, im_);

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
              net.meta.classes.description{best}, best, bestScore), ...
      'Interpreter', 'none') ;
figure(2);
plot(scores);

imsz = net.meta.normalization.imageSize;



%generate white noise image;
im0 = generateWhiteNoisImage(imsz);

%Code outline for gradient descent across each layer. 
%Please edit the outline as implementation changes.
%im0 should be updated until it matches the content of the image
%above applied to the trained network

step = 0.1;
Niterations = 20;
gradient = 1;
for i = 1:Niterations
  %calculate error by back-propagation
  for layer = fliplr(1:nLayers)
    type = net.layers{layers}.type;
    %for each layer type, write a function that performs 
    %gradient descent based on input parameters.
    %'params' isn't defined:please edit to suit the actual function
    %you write
    %input parameters are for example padding, size of filter
    %for convolutional layer. 
    if strcmp(type, 'conv')
      %{ 
      x = res{i}.x;
      sz = size(x);
      %vectorize
      x = permute(x, [2 1 sz(3)]);
      x = reshape(x, [length(sz) 1]);
      %}
      gradient = convGD(params) * gradient; 
    elseif strcmp(type, 'softmax')
      gradient = softmaxGD(params) * gradient;
    elseif strcmp(type, 'relu')
      gradient = reluGD(params) * gradient;
    elseif strcmp(type, 'pool')
      gradient = reluGD(pool) * graident;
    end 
    %update im0
    im0 = im0 - step*gradient;
  end
end
