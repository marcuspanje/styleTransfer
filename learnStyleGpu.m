setup;

desiredLayers = [11 27];
%desiredLayers = [27];
desiredLayerWeights = gpuArray([1/2 1/2]);
%desiredLayerWeights = gpuArray([1]);
layerI = 1;

loadNet = 1;
if loadNet
    net = load('vgg-face.mat');
    net.layers(max(desiredLayers)+1:end) = [];
    net = vl_simplenn_tidy(net);
    net = vl_simplenn_move(net, 'gpu');
end
avgImg = net.meta.normalization.averageImage;

%images must be 244x244

% load content image
%im = imread('img/khan.jpg');
%im_ = bsxfun(@minus, single(im), avgImg);
%imContent = vl_simplenn(net, im_);

% load style image
im = imread('img/vg5.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imStyle = vl_simplenn(net, gpuArray(im_));

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, gpuArray(im0_));

disp('generating new image');

Niterations = 1000;

%std gradient descent params
step = 0.01;      %gradient des step size

%grad descent with momentum params
gamma = 0.7; 
v = 0;

%calculate error by back-propagation
%desiredLayers = [3 8 13 20 27];

%record error every [plotInterval] timesteps
plotInterval = 1;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;
imUpdateGpu = gpuArray(imNew(1).x);
zerosGpu = zeros(size(imNew(1).x), 'gpuArray');
prev_error = gpuArray(0);

%ADAM parameters:
mPrev = zerosGpu;
vPrev = zerosGpu;
beta1 = gpuArray(0.9);
beta2 = gpuArray(0.999);
epsilon = 1e-8;

for iter = 1:Niterations
    
    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradSum = zerosGpu;
    style_error = gpuArray(0);

    %grad descent with adadelta params
    gradSum = zeros(size(im));
    gradSumEps = 1e-5;
    gammaAda = 0.2;
    gradPrev2 = zeros(size(im));
    
    for layerI = 1:length(desiredLayers);
        l = desiredLayers(layerI);
        w_l = desiredLayerWeights(layerI);
        netI.layers = net.layers(1:l);
        [h0,w0,d0] = size(imNew(l+1).x);
        nParams = h0*w0*d0;
        F = to2D(imNew(l+1).x);
        G = Gram(F);
        A = Gram(to2D(imStyle(l+1).x));
        gradNext = (1/nParams^2)*(F'*(G-A))';
        gradNext(find(F<0))=0;
        gradNext = single(toND(gradNext,h0,w0));
        %apply backward pass
        imNewI = vl_simplenn(netI, imNew(1).x, gradNext, imNew, 'SkipForward', true);
        gradSum = gradSum + w_l*imNewI(1).dzdx; 
        style_error = style_error + w_l*LayerStyleError(G, A, nParams);
    end

    %standard update
    %if iter > 1 && prev_style_error <= style_error
    %  step = step/2
    %end
    %imNew(1).x = imNew(1).x - step*gradSum;

    %momentum update
    %if iter > 1 && prev_style_error <= style_error
    %  gamma = (gamma + 1)/2;
    %end
    %v = gamma*v + step*gradSum; 
    %imNew(1).x = imNew(1).x - v;

    %ADAM updatee
    gradSum2 = gradSum.^2;
    m = (beta1*mPrev + (1-beta1)*gradSum);
    v = beta2*vPrev + (1-beta2)*gradSum2;
    mPrev = m;
    vPrev = v;
    m = m/(1-beta1.^iter);
    v = v/(1-beta2.^iter);
    update = m.*step./(sqrt(v)+epsilon);
    imNew(1).x = imNew(1).x - update;

    imNew = vl_simplenn(net, imNew(1).x);
    prev_style_error = style_error;

    % record error if desired
    if iter == plotIndices(plotI) 
      err(plotI) =  gather(style_error);
      disp(sprintf('iteration %03d, style_error: %.2f', iter, style_error));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end
    end
end % for each iteration

imNewDisp= uint8(gather(bsxfun(@plus, imNew(1).x, avgImg)));
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
