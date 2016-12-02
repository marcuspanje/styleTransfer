%setup is untracked by git, as it is different for all users. 
%create setup.m with one line:
%run('[path to matconvnet files]/vl_setupnn');
setup;

%desired layers for style learning
%desiredLayers = gpuArray([3 8 13 20 27]);
desiredLayers = gpuArray([3 8 13]);
desiredLayerWeights = gpuArray([1/3 1/3 1/3]);
%desiredLayerWeights = gpuArray([1/5 1/5 1/5 1/5 1/5]);
%layer for content learning
L = 13;

%load trained network
if exist('net') ~= 1 
    disp('loading network');
    %net = load('imagenet-vgg-verydeep-16.mat');
    net = load('vgg-face.mat');
    net.layers(max(desiredLayers)+1:end) = [];
    net = vl_simplenn_tidy(net);
    net = vl_simplenn_move(net, 'gpu');
end
avgImg = net.meta.normalization.averageImage;

%images must be 244x244
% load content image
im = imread('img/khan.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imContent = vl_simplenn(net, gpuArray(im_));

% load style image
im = imread('img/vg5.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imStyle = vl_simplenn(net, gpuArray(im_));

%generate white noise image;
imsz = net.meta.normalization.imageSize(1:3);
im0 = single(generateWhiteNoiseImage(imsz));
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, gpuArray(im0_));

disp('generating new image');

Niterations = 5000;
%step decreases by annealFactor every so often
annealFactor = gpuArray(0.7);

zerosGpu = zeros(size(imNew(1).x), 'gpuArray');
%std gradient descent params
step = gpuArray(1);      %gradient des step size

%grad descent with momentum params
gamma = 0.7; 
v = zerosGpu;

%calculate error by back-propagation

%record error every [plotInterval] timesteps
prevError = 0;
plotInterval = 5;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
errContent = err;
errStyle = err;
plotI = 1;

gradWeights = gpuArray([1 1 1000 1000]);
gradWeights = gradWeights ./ sum(gradWeights);

%ADAM parameters:
mPrev = zerosGpu;
vPrev = zerosGpu;
beta1 = gpuArray(0.9);
beta2 = gpuArray(0.999);
beta1Power = gpuArray(1);
beta2Power = gpuArray(1);
epsilon = gpuArray(1e-8);


for iter = 1:Niterations
    
    %gradient for style:
    [gradStyle, style_error] = computeGradStyle(net, imNew, imStyle, ... 
        desiredLayers, desiredLayerWeights);

    %gradient for Content
    diffContent = imNew(L+1).x - imContent(L+1).x;
    gradNext = diffContent;
    gradNext(imNew(L+1).x < 0) = 0;
    %back prop with our functions
    gradContent = backProp(net, L, imNew, gradNext);    
    %imNewI = vl_simplenn(net, imNew(1).x, gradNext, imNew, 'SkipForward', true);
    %gradContent = imNewI(1).dzdx;


%The following regularization errors are described in
%"Understanding Deep Image Representations by Inverting Them " 
%by Mahendran et. al, Univ. Oxford

    %error to limit size of values (avoid too large pixel values)
    %L(x(i,j)) = 0.5*x(i,j)^2 
    gradSize = imNew(1).x;

    %error to limit variation between pixels (like low pass filter)
    %L(x(i,j)) = 0.5( (x(i,j+1)-x(i,j))^2 + (x(i+1,j)-x(i,j))^2 )
    shiftRight = zerosGpu;
    shiftDown = shiftRight;
    shiftRight(:,1:end-1,:) = imNew(1).x(:,1:end-1,:)-imNew(1).x(:,2:end,:);
    shiftDown(1:end-1,:,:) = imNew(1).x(1:end-1,:,:)-imNew(1).x(2:end,:,:);
    gradVariation = shiftRight + shiftDown;


    grad = gradWeights(1)*gradStyle + gradWeights(2)*gradContent +  ...
      gradWeights(3)*gradSize + gradWeights(4)*gradVariation;

    %standard update
    %imNew(1).x = imNew(1).x - step*grad;

    %momentum update
    %v = gamma*v + step*grad; 
    %imNew(1).x = imNew(1).x - v;

    %ADAM updatee

    m = (beta1*mPrev + (1-beta1)*grad);
    v = beta2*vPrev + (1-beta2)*(grad.^2);
    mPrev = m;
    vPrev = v;
    beta1Power = beta1Power * beta1;
    beta2Power = beta2Power * beta2;
    m = m/(1-beta1Power);
    v = v/(1-beta2Power);
    update = m.*step./(sqrt(v)+epsilon);
    imNew(1).x = imNew(1).x - update;


    %reapply image on network
    imNew = vl_simplenn(net, imNew(1).x);

    % record error if desired
    if iter == plotIndices(plotI) 
      errContentI = 0.5*sumsqr(diffContent);
      errStyleI = style_error; 
      errSizeI = 0.5*sumsqr(imNew(1).x);
      errVariationI = 0.5*(sumsqr(shiftRight) + sumsqr(shiftDown));

      errTotalI = gradWeights(1)*errStyleI + gradWeights(2)*errContentI + ... 
        gradWeights(3)*errSizeI  + gradWeights(4)*errVariationI;

      err(plotI) = gather(errTotalI);
      disp(sprintf('iteration %03d, error: %.2f', iter, errTotalI));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end

      %anneal step
      if iter > 50 && prevError < errTotalI
        step = annealFactor*step
      end

      prevError = errTotalI;

    end
end % for each iteration

max(max(imNew(1).x))
min(min(imNew(1).x))
plotter;


