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
im0 = generateWhiteNoiseImage(imsz);
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imR = vl_simplenn(net, im0_);

disp('generating new image');
%reference layer
figure(1);
L = 2;
gradNext = single(imR(L+1).x - imC(L+1).x);
step = 0.01;%gradient des step size
Niterations = 20;
for iter = 1:Niterations
    %calculate error by back-propagation
    gradNext = imR(L+1).x - imC(L+1).x;
    if mod(iter, 2) == 0
        err = gradNext.^2;
        err = sum(sum(sum(err)));
        disp(sprintf('iteration %03d, err: %d', iter, err));
    end
    
    for layer = fliplr(1:L)
        
        
        type = net.layers{layer}.type;
        szYprev = size(imR(layer).x);
        grad = zeros(szYprev);
        
        if strcmp(type, 'conv')
            weights = net.layers{layer}.weights{1};
            pad = net.layers{layer}.pad;
            stride = net.layers{layer}.stride;
            Yprev = imR(layer).x;
            
            [grad,~,~] = vl_nnconv(Yprev, weights, [], gradNext, ...
                'pad', pad, 'stride', stride);
            
        elseif strcmp(type, 'softmax')
            %gradient = softmaxGD(params) * gradient;
        elseif strcmp(type, 'relu')
            
            
            
            %DZDX = VL_NNRELU(X, DZDY)
            
            %        DZDY = gradNext;
            
            %        DZDY = zeros(szYprev(1), szYprev(2), szYprev(3));
            
            for i = 1:szYprev(1)
                for j = 1:szYprev(2)
                    for k = 1:szYprev(3)
                        if(imR(layer).x(i, j, k) < 0)
                            grad(i, j, k) = gradNext(i, j, k);
                        end
                    end
                end
            end
            
            %grad = vl_nnrelu(imR(L+1).x, DZDY);
            
        elseif strcmp(type, 'pool')
            %gradient = reluGD(pool) * graident;
            Yprev = imR(layer).x;
            pool = net.layers{layer}.pool;
            grad = vl_nnpool(Yprev,pool,gradNext);
        end
        gradNext = single(grad);
        
    end %for each layer
    
    imR(1).x = imR(1).x - step*grad;
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
