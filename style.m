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
imC = vl_simplenn(net, im_);

% load style image
im = imread('img/vg5.jpg');
im_ = bsxfun(@minus, single(im), avgImg);
imS = vl_simplenn(net, im_);

%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imR = vl_simplenn(net, im0_);

disp('generating new image');
%reference layer
L = 20;


step = 0.001;      %gradient des step size
Niterations = 120;
%calculate error by back-propagation
for iter = 1:Niterations
    
    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradSum = zeros(size(imR(1).x));
    for l=1:L
        
        [h0,w0,d0] = size(imR(l+1).x);
        F = to2D(imC(l+1).x);
        G = Gram(F);
        A = Gram(to2D(imS(l+1).x));
        gradNext = (1/(h0*w0*d0)^2)*(F'*(G-A))';
        gradNext(find(F<0))=0;
        gradNext = single(toND(gradNext,h0,w0));
        
        % BP
        for layer = fliplr(1:l)
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
                
            elseif strcmp(type, 'softmax')
                %gradient = softmaxGD(params) * gradient;
            elseif strcmp(type, 'relu')
                %DZDX = VL_NNRELU(X, DZDY)
                %grad = vl_nnrelu(imR(layer).x, gradNext);
                grad = vl_nnrelu(Yprev, gradNext);
                
            elseif strcmp(type, 'pool')
                %gradient = poolGD(pool) * graident;
                pool = net.layers{layer}.pool;
                stride = net.layers{layer}.stride;
                grad = vl_nnpool(Yprev,pool,gradNext, 'stride', stride);
                
            end
            
            gradNext = single(grad);
            
        end %for each layer
        
        gradSum = single(gradSum + grad);
        
    end %l - for suming L_layer
    %------------------------------------------
%     if mod(iter, 2) == 0
        err = gradSum.^2;
        err = sum(err(:));
        disp(sprintf('iteration %03d, err: %d', iter, err));
%     end
    %       --------------------------------------------
    
    imR(1).x = imR(1).x - step*gradSum;
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
