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

Niterations = 50;

%std gradient descent params
step = 0.000005;      %gradient des step size

%grad descent with momentum params
gamma = [0.5 0.9 0.95 0.97 0.99 0];
%gamma = 0.9; 
v = 0;
momI = 1;

%calculate error by back-propagation
%desiredLayers = [3 8 13 20 27];
desiredLayers = [20 27];
nLayers = length(desiredLayers);
desiredLayerWeights = gpuArray([1/2 1/2]);

%record error every [plotInterval] timesteps
plotInterval = 1;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;
imUpdateGpu = gpuArray(imNew(1).x);
zerosGpu = zeros(size(imNew(1).x), 'gpuArray');
prev_error = gpuArray(0);
for iter = 1:Niterations
    
    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradSum = zerosGpu;
    count = 1;
    style_error = gpuArray(0);
    t1 = 0; t2 = 0;
    for l = desiredLayers
        
        tic;
        w_l = desiredLayerWeights(count);
        count = count + 1;
        [h0,w0,d0] = size(imNew(l+1).x);
        nParams = h0*w0*d0;
        F = to2D(imNew(l+1).x);
        G = Gram(F);
        A = Gram(to2D(imStyle(l+1).x));
        gradNext = (1/nParams^2)*(F'*(G-A))';
        gradNext(find(F<0))=0;
        gradNext = single(toND(gradNext,h0,w0));
        t1 = t1 + toc;
        %disp('calculated style error');
        tic;
        % BP
        for layer = fliplr(1:l)
            type = net.layers{layer}.type;
            szYprev = size(imNew(layer).x);
            grad = zeros(szYprev, 'gpuArray');
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
        %disp('done back prop');
        gradSum = single(gradSum + w_l*grad);
        
        %Error for layer l, equation 4
        style_error = style_error + w_l*LayerStyleError(G, A, nParams);
        t2 = t2 + toc;
        
    end %l - for suming L_layer
   
    tic; 
    %standard update
    %imNew(1).x = imNew(1).x - step*gradSum;
        
    %momentum update
    %update gamma depending on error size
    v = gamma(momI)*v - step*gradSum;
    imNew(1).x = imNew(1).x + v;
    if prev_error ~= 0 && style_error >= prev_error
      if momI < length(gamma)
        momI = momI+1; 
      else 
        %decrease step size
        step = step/2;
      end
    end

    %reapply network on image
    imNew = vl_simplenn(net, imNew(1).x);
    t3 = toc;

    % record error if desired
    if iter == plotIndices(plotI) 
      err(plotI) =  gather(style_error);
      disp(sprintf('iteration %03d, style_error: %.2f', iter, style_error));
      %disp(sprintf('timing: init: %.2f, back prop: %.2f, upate: %.2f\n', ...
    %    t1/nLayers, t2/nLayers, t3));        
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end
    end
    prev_error = style_error;
end % for each iteration
toc;

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
