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
imC = vl_simplenn(net, im_);

%generate white noise image;
imsz = net.meta.normalization.imageSize;
%im0 = generateWhiteNoiseImage(imsz);
im0 = imread('img/whitenoise.jpg');
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imR = vl_simplenn(net, im0_);

disp('generating new image');
%reference layer
L = 29;
%gradSz = size(imR(L+1).x);
nParams = sum(size(imR(L+1).x));
step = 0.1;%gradient des step size
gamma = 0.6; %momentum
plotInterval = 3;
Niterations = 30;
%err is the index of iteratoin to plot
%error is computed and filled to err in-place
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;
v = 0;
gradSum = zeros(size(im));
gradSumEps = 1e-5;
gammaAda = 0.8;
gradPrev2 = zeros(size(im));
for iter = 1:Niterations
    %calculate error by back-propagation
    gradNext = imR(L+1).x - imC(L+1).x;
    if iter == plotIndices(plotI) 
      err(plotI) = sum(sum(sum(gradNext.^2))) ./ nParams;
      disp(sprintf('iteration %03d, err: %.1f', iter, err(plotI)));
      plotI = plotI + 1;
    end
%{
    if mod(iter, 5) == 0
        err = gradNext.^2;
        err = sum(sum(sum(err)));
        disp(sprintf('iteration %03d, err: %d', iter, err));
    end
%}
    
    %for layer = fliplr(1:L)
    for layer = fliplr(1:L)
        type = net.layers{layer}.type;
        szYprev = size(imR(layer).x);
        grad = zeros(szYprev);
        Yprev = single(imR(layer).x);

        if strcmp(type, 'conv')
            weights = net.layers{layer}.weights{1};
            bias = net.layers{layer}.weights{2};
            pad = net.layers{layer}.pad;
            stride = net.layers{layer}.stride;
            
            [grad,~,~] = vl_nnconv(Yprev, weights, bias, gradNext, ...
                'pad', pad, 'stride', stride);
            
        elseif strcmp(type, 'relu')
            %DZDX = VL_NNRELU(X, DZDY)
            %grad = vl_nnrelu(imR(layer).x, gradNext);
            grad = vl_nnrelu(Yprev, gradNext);
            
        elseif strcmp(type, 'pool')
            pool = net.layers{layer}.pool;
            stride = net.layers{layer}.stride;
            grad = vl_nnpool(Yprev,pool,gradNext, 'stride', stride);
            
        end
        gradNext = single(grad);
        
    end %for each layer
    standard update
    imR(1).x = imR(1).x - step*grad;


    %momentum update
    %v = gamma*v + step*grad; 
    %imR(1).x = imR(1).x - v;



    %adaGrad update
%{
    grad2 = grad.^2;
    gradSum = gammaAda*gradPrev2 + (1-gammaAda)*grad2;
    newStep = step./sqrt(gradSum+gradSumEps);
    imR(1).x = imR(1).x - newStep.*grad;
     
    gradPrev2 = grad2;
%}


    %reapply network on image
    imR = vl_simplenn(net, imR(1).x);
end % for each iteration

imRDisp= uint8(bsxfun(@plus, imR(1).x, avgImg));
figure(1);
subplot(121);
imshow(im); %original image
title('reference');
subplot(122);
imshow(imRDisp);
title('generated');

figure(2);
plot(plotIndices, err, 'x-');
xlabel('iterations');
ylabel('error');
