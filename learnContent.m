%%% inputFace above %%%
%replace path with location of vl_setupnn from matlabconvnet package
%run('~/code/matconvnet-1.0-beta21/matlab/vl_setupnn');
%load weights of the trained vgg-face network
%this repo does not store the mat file. It can be obtianed from:
%http://www.vlfeat.org/matconvnet/pretrained/

setup;
loadNet = 0;
if loadNet
    net = load('vgg-face.mat');
    net = vl_simplenn_tidy(net);
end
avgImg = net.meta.normalization.averageImage;

%images must be 244x244
im = imread('img/khan.jpg');
%content
im_ = bsxfun(@minus, single(im), avgImg) ;
imContent = vl_simplenn(net, im_);

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, im0_);

disp('generating new image');
L = 27;
Niterations = 20;

nParams = sum(size(imNew(L+1).x));

%record error every [plotInterval] timesteps
plotInterval = 3;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;

%std gradient descent params
step = 0.01;
%grad descent with momentum params
gamma = 0.6; 
v = 0;
%grad descent with adadelta params
gradSum = zeros(size(im));
gradSumEps = 1e-5;
gammaAda = 0.2;
gradPrev2 = zeros(size(im));

for iter = 1:Niterations
    %calculate error by back-propagation
    gradNext = imNew(L+1).x - imContent(L+1).x;
    if iter == plotIndices(plotI) 
      err(plotI) = sum(sum(sum(gradNext.^2))) ./ nParams;
      disp(sprintf('iteration %03d, err: %.1f', iter, err(plotI)));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end
    end
    
    for layer = fliplr(1:L)
        type = net.layers{layer}.type;
        szYprev = size(imNew(layer).x);
        grad = zeros(szYprev);
        Yprev = single(imNew(layer).x);

        if strcmp(type, 'conv')
            weights = net.layers{layer}.weights{1};
            bias = net.layers{layer}.weights{2};
            pad = net.layers{layer}.pad;
            stride = net.layers{layer}.stride;
            
            [grad,~,~] = vl_nnconv(Yprev, weights, bias, gradNext, ...
                'pad', pad, 'stride', stride);
            
        elseif strcmp(type, 'relu')
            %DZDX = VL_NNRELU(X, DZDY)
            grad = vl_nnrelu(Yprev, gradNext);
            
        elseif strcmp(type, 'pool')
            pool = net.layers{layer}.pool;
            stride = net.layers{layer}.stride;
            grad = vl_nnpool(Yprev,pool,gradNext, 'stride', stride);
            
        end
        gradNext = single(grad);
        
    end %for each layer

    %standard update
    imNew(1).x = imNew(1).x - step*grad;

    %momentum update
    %v = gamma*v + step*grad; 
    %imNew(1).x = imNew(1).x - v;

    %adaGrad update
    %grad2 = grad.^2;
    %gradSum = gammaAda*gradPrev2 + (1-gammaAda)*grad2;
    %newStep = step./sqrt(gradSum+gradSumEps);
    %imNew(1).x = imNew(1).x - newStep.*grad;
    %gradPrev2 = grad2;


    %reapply network on image
    imNew = vl_simplenn(net, imNew(1).x);
end % for each iteration

imNewDisp= uint8(bsxfun(@plus, imNew(1).x, avgImg));
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
