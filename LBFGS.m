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

Niterations = 3000;
annealFactor = gpuArray(0.8);

%std gradient descent params
step = gpuArray(0.1);      %gradient des step size

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

[h,w,d] = size(imNew(1).x);
% l-bfgs parameters
m =gpuArray(5);
s = zeros(h*w*d,m,'gpuArray');
y = zeros(h*w*d,m,'gpuArray');
alpha = zeros(1,m,'gpuArray');
beta = zeros(1,m,'gpuArray');
rou = zeros(1,m,'gpuArray');
gradPrev = zeros(h*w*d,1,'gpuArray');

for iter = 1:Niterations    
    
   
    if(iter > 2)
        gradPrev = grad1d;
    end %if
    
    % ============ calculate grad with BP ==============================   
    % gradient for style:
    % recompute gradNext
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    gradStyleSum = zerosGpu;
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

        gradStyleSum = gradStyleSum + w_l*gradStyle; 
        style_error = style_error + w_l*sumsqr(diffStyle)/(4*nParams^2);
    end % for each layer

    %gradient for Content
    diffContent = imNew(L+1).x - imContent(L+1).x;
    gradNext = diffContent;
    gradNext(imNew(L+1).x < 0) = 0;
    %back prop with our functions
    gradContent = backProp(net, L, imNew, gradNext);    

    %error to constraint size of values
    gradSize = imNew(1).x;
    errSizeI = 0.5*sumsqr(imNew(1).x);
    grad = styleWeight*gradStyleSum + contentWeight*gradContent +  sizeWeight*gradSize;
% ====================================================================================
    
% ============== l-bfgs update =======================================================   
   
    grad1d = mtx2vec(grad);
    imNew1D = mtx2vec(imNew(1).x);
    imPrev = imNew1D;
    
%     % adjust step size
%     if(iter > 1 && prev_style_error <= style_error)
%         step = step/2
%     end
        
    if(iter==1)
        %standard update
        imNew(1).x = imNew(1).x - step*grad;  
        s(:,1) = mtx2vec(imNew(1).x) - imPrev;
    end %if
        
    if(iter >= 2 && iter<=(m+1))
        q = grad1d;
        y(:,iter-1) = grad1d - gradPrev;
        rou(iter-1) = 1/(y(:,iter-1)'*s(:,iter-1));
      
        for k = (iter-1):-1:1
            alpha(k) = rou(k)*s(:,k)'*q;
            q = q - alpha(k)*y(:,k);
            
        end %k
        H0 = (s(:,iter-1)'*y(:,iter-1))/(y(:,iter-1)'*y(:,iter-1));
        z = H0*q;
        for k=1:iter-1
            beta(k) = rou(k)*y(:,k)'*z;
            z = z + s(:,k)*(alpha(k)-beta(k));
        end %k
        
        imNew(1).x = imNew(1).x - step*vec2mtx(z,h,w,d);
        s(:,iter) = mtx2vec(imNew(1).x) - imPrev;
    end %if
    
    
    if(iter > m+1)
        y(:,1:m-1) = y(:,2:m);
        y(:,m) = grad1d - gradPrev;
        rou(1:m-1) = rou(2:m);
        rou(m) = 1/(y(:,m)'*s(:,m));
        q = grad1d;
        
        for k=m:-1:1
            alpha(k) = rou(k)*s(:,k)'*q;
            q = q - alpha(k)*y(:,k);            
        end %k
        
        H0 = s(:,m)'*y(:,m)/(y(:,m)'*y(:,m));
        z = H0*q;
        
        for k=1:m
            beta(k) = rou(k)*y(:,k)'*z;
            z = z + s(:,k)*(alpha(k)-beta(k));            
        end %k
        imNew(1).x = imNew(1).x - step*vec2mtx(z,h,w,d);
        s(:,1:m-1) = s(:,2:m);
        s(:,m) = mtx2vec(imNew(1).x) - imPrev;
    end %if
% =======================================================================

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

img = im2uint8(imNew(1).x);
img = gather(img);
save('img1.mat','img');

plotter;


