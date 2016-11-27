%%% inputFace above %%%
%replace path with location of vl_setupnn from matlabconvnet package
%run('~/code/matconvnet-1.0-beta21/matlab/vl_setupnn');
%load weights of the trained vgg-face network
%this repo does not store the mat file. It can be obtianed from:
%http://www.vlfeat.org/matconvnet/pretrained/

setup;
loadNet = 0;
L = 27;

if loadNet
    net = load('vgg-face.mat');
    net.layers(L+1:end) = [];
    net = vl_simplenn_tidy(net);
    net = vl_simplenn_move(net, 'gpu');
end
avgImg = net.meta.normalization.averageImage;

%images must be 244x244
im = imread('img/khan.jpg');
%content
im_ = bsxfun(@minus, single(im), avgImg) ;
imContent = vl_simplenn(net, gpuArray(im_));

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, gpuArray(im0_));

disp('generating new image');
Niterations = 30;

nParams = sum(size(imNew(L+1).x));

%record error every [plotInterval] timesteps
plotInterval = 2;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;

zerosGpu = zeros(size(imNew(1).x), 'gpuArray');

%std gradient descent params
%step = 0.1;
step = 1;
%grad descent with momentum params
%gamma = 0.6; 
%v = zerosGpu;
%ADAM parameters:
mPrev = zerosGpu;
vPrev = zerosGpu;
beta1 = gpuArray(0.9);
beta2 = gpuArray(0.999);
beta1Power = 1;
beta2Power = 1;
epsilon = 1e-8;

for iter = 1:Niterations
    %calculate error by back-propagation
    gradNext = imNew(L+1).x - imContent(L+1).x;
    %back prop with our functions
    grad = backProp(net, L, imNew, gradNext);    

%%% use vl_simplenn lib function to compute grad:
    %imNew = vl_simplenn(net, imNew(1).x, gradNext, imNew, 'SkipForward', true);
    %grad = imNew.dzdx 
    %%%%

    %%% standard update
    %imNew(1).x = imNew(1).x - step*(grad);
    %%%

    %%% momentum update
    %v = gamma*v + step*grad; 
    %imNew(1).x = imNew(1).x - v;
    %%%

    %ADAM updatee
    m = (beta1*mPrev + (1-beta1)*grad);
    v = beta2*vPrev + (1-beta2)*(grad.^2);
    mPrev = m;
    vPrev = v;
    beta1Power = beta1Power * beta1;
    beta2Power = beta2Power * beta2;
    m = m/(1-beta1Power);
    v = v/(1-beta2Power);
    %stepCurrent = step/sqrt(iter);
    update = m.*step./(sqrt(v)+epsilon);
    imNew(1).x = imNew(1).x - update;

    %reapply network on image
    imNew = vl_simplenn(net, imNew(1).x);

    %plot if desired
    if iter == plotIndices(plotI) 
      err(plotI) = gather(sum(sum(sum(gradNext.^2)))) ./ nParams;
      disp(sprintf('iteration %03d, err: %.1f', iter, err(plotI)));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end
    end
end % for each iteration

imNewDisp= uint8(bsxfun(@plus, gather(imNew(1).x), avgImg));
imwrite(imNewDisp, 'img.jpg');
figure(1);
subplot(121);
imshow(im); %original image
title('reference');
subplot(122);
imshow(imNewDisp);
title('generated');

figure(2);
plot(plotIndices, err, 'x-');
xlabel('iterations');
ylabel('error');
