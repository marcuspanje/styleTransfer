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

% load content image
im = imread('img/khan.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imContent = vl_simplenn(net, im_);

% load style image
im = imread('img/vg5.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imStyle = vl_simplenn(net, im_);

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, im0_);

disp('generating new image');

Niterations = 25;

%std gradient descent params
step = 0.000001;
%grad descent with momentum params
%gamma = 0.7; 
v = 0;

%calculate error by back-propagation
desiredLayers = [3 8 13 20 27];
desiredLayerWeights = [1 1/2 1/2 1/4 1/5];
%desiredLayers = [27];
%desiredLayerWeights = [1];


%record error every [plotInterval] timesteps
plotInterval = 1;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;

for iter = 1:Niterations
    

    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradSum = zeros(size(imNew(1).x));
    count = 1;
    errSum = 0;
    for l = desiredLayers
        w_l = desiredLayers(count);
        count = count + 1;
        [h0,w0,d0] = size(imNew(l+1).x);
        F = to2D(imNew(l+1).x);
        G = Gram(F);
        A = Gram(to2D(imStyle(l+1).x));
        gradNext = (1/(h0*w0*d0)^2)*(F'*(G-A))';
        gradNext(find(F<0))=0;
        gradNext = single(toND(gradNext,h0,w0));
        sqGA = (G-A).^2;
        errSum = errSum + w_l/(4*w0*h0) * sum(sum(sqGA));
        % BP
        for layer = fliplr(1:l)
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
        
        gradSum = single(gradSum + w_l*grad);
        
    end %l - for suming L_layer
    %------------------------------------------
%     if mod(iter, 2) == 0
       % err = gradSum.^2;
       % err = sum(err(:));
       % disp(sprintf('iteration %03d, err: %d', iter, err));
%     end
    %       --------------------------------------------
    
    %standard update
    imNew(1).x = imNew(1).x - step*gradSum;

    %momentum update
    %v = gamma*v + step*gradSum;
    %imNew(1).x = imNew(1).x - v;

    %reapply network on image
    imNew = vl_simplenn(net, imNew(1).x);

    % record error if desired
    if iter == plotIndices(plotI) 
      err(plotI) =  errSum;
      disp(sprintf('iteration %03d, err: %.1f', iter, err(plotI)));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end
    end
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
