setup;

desiredLayers = gpuArray([3 8 13 20 27]);
desiredLayerWeights = gpuArray([1/5 1/5 1/5 1/5 1/5]);
L = 27;

loadNet = 0;
if loadNet
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
imsz = net.meta.normalization.imageSize;
im0 = single(generateWhiteNoiseImage(imsz));
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, gpuArray(im0_));
nParamsContent = sum(size(imNew(L+1).x));

disp('generating new image');

Niterations = 10000;
annealFactor = gpuArray(0.8);

%std gradient descent params
step = gpuArray(1);      %gradient des step size

%grad descent with momentum params
gamma = 0.7; 
v = 0;

%calculate error by back-propagation
%desiredLayers = [3 8 13 20 27];

%record error every [plotInterval] timesteps
prevError = 0;
plotInterval = 5;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
errContent = err;
errStyle = err;
plotI = 1;
zerosGpu = zeros(size(imNew(1).x), 'gpuArray');

styleWeight = gpuArray(0.001); % weightage of style
contentWeight = gpuArray(1);
sizeWeight = gpuArray(0.33);
totalWeight = styleWeight + contentWeight + sizeWeight;
styleWeight = styleWeight/totalWeight;
contentWeight = contentWeight/totalWeight;
sizeWeight = sizeWeight/totalWeight;
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
    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradSum = zerosGpu;
    style_error = gpuArray(0);

    for layerI = 1:length(desiredLayers);
        l = desiredLayers(layerI);
        w_l = desiredLayerWeights(layerI);
        netI.layers = net.layers(1:l);
        [h0,w0,d0] = size(imNew(l+1).x);
        nParams = h0*w0*d0;
        F = to2D(imNew(l+1).x);
        G = Gram(F);
        A = Gram(to2D(imStyle(l+1).x));
        diffStyle = G-A;
        gradNext = (1/nParams^2)*(F'*(diffStyle))';
        gradNext(F<0)=0;
        gradNext = single(toND(gradNext,h0,w0));
        %apply backward pass
        gradStyle = backProp(net, l, imNew, gradNext);    

        gradSum = gradSum + w_l*gradStyle; 
        style_error = style_error + w_l*sumsqr(diffStyle)/(4*nParams^2);
    end

    %gradient for Content
    diffContent = imNew(L+1).x - imContent(L+1).x;
    gradNext = diffContent;
    gradNext(imNew(L+1).x < 0) = 0;
    %back prop with our functions
    gradContent = backProp(net, L, imNew, gradNext);    

    %error to constraint size of values
    gradSize = imNew(1).x;
    errSizeI = 0.5*sumsqr(imNew(1).x);
    

    grad = styleWeight*gradSum + contentWeight*gradContent +  sizeWeight*gradSize;

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


    imNew = vl_simplenn(net, imNew(1).x);

    errContentI = sumsqr(diffContent)/2;
    errStyleI = style_error; 
    errTotalI = contentWeight*errContentI + styleWeight*errStyleI + sizeWeight*errSizeI;

    % record error if desired
    if iter == plotIndices(plotI) 
      errContent(plotI) = gather(errContentI);
      errStyle(plotI) = gather(errStyleI);
      err(plotI) = gather(errTotalI);
      disp(sprintf('iteration %03d, error: %.2f', iter, errTotalI));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end

      %anneal step
      if iter > 500 && prevError < errTotalI
        step = annealFactor*step;
      end
      prevError = errTotalI;

    end
end % for each iteration

plotter;


