%setup is untracked by git, as it is different for all users. 
%create setup.m with one line:
%run('[path to matconvnet files]/vl_setupnn');
setup;

%desired layers for style learning
desiredLayers = gpuArray([3 8 13 20 27]);
desiredLayerWeights = gpuArray([1/5 1/5 1/5 1/5 1/5]);
lenDesiredLayers = length(desiredLayers);
%layer for content learning
L = 27;

%1 to use gradient style combination
%0 to use eigenface style combination
useGradientStyleCombination = 1;

%load trained network
if exist('net') ~= 1 
    disp('loading network');
    net = load('vgg-face.mat');
    net.layers(max(desiredLayers)+1:end) = [];
    net = vl_simplenn_tidy(net);
    net = vl_simplenn_move(net, 'gpu');
end
avgImg = net.meta.normalization.averageImage;

%images must be 244x244
% load content image
contentImage = 'img/others/chow.jpg';
im = imread(contentImage);
imContentScaled = bsxfun(@minus, single(im), avgImg);
imContentNet = vl_simplenn(net, gpuArray(imContentScaled));
imContentMean = gpuArray(mean(mean(im)) - avgImg); 
%only keep Lth layer. 
imContentNetL = imContentNet(L+1).x;
clear imContentNet;

% load style images
styleImageList = {'img/vg/vg1.jpg', 'img/vg/vg2.jpg', 'img/vg/vg3.jpg', ...
    'img/vg/vg4.jpg', 'img/vg/vg5.jpg'};
numStyleImages = length(styleImageList);
GramLayers = cell(lenDesiredLayers, 1);

%for each style image, store only Gram matrices of desired layers
%GramLayers stores the gram matrixes of desired layers of a single image
%GramStyleList is an array of structures containing the GramLayers cells of each image
%start from back to avoid array resizing of GramStyleList
for i = fliplr(1 : numStyleImages)

	im = imread(styleImageList{i});
	imStyleScaled = bsxfun(@minus, single(im), avgImg);
    imStyleNet = vl_simplenn(net, gpuArray(imStyleScaled));

    for layerI = 1:lenDesiredLayers; 
        l = desiredLayers(layerI);
        GramLayers{layerI} = Gram(to2D(imStyleNet(l+1).x)); 
    end

    GramStyleList(i).GramLayers = GramLayers;

    %display only style image 1
    if i == 1
        imStyleScaled1 = imStyleScaled;
    end
end

clear imStyleNet;
EigenGramLayers = cell(lenDesiredLayers, 1);

for layerI = 1:lenDesiredLayers

  l = desiredLayers(layerI);
  sz = size(GramStyleList(1).GramLayers{layerI});
  GramVectorizedMat = zeros(sz(1)*sz(2), numStyleImages, 'gpuArray');
  %Each column of GramVectorizedMat is the vectorized gram matrix
  % for a layer in an image
  for i = 1:numStyleImages
    GramVectorizedMat(:, i) = ...
        reshape(GramStyleList(i).GramLayers{layerI}, sz(1)*sz(2), 1);
  end
  meanGram = mean(GramVectorizedMat, 2); 
  GramVectorizedMat = bsxfun(@minus, GramVectorizedMat, meanGram);
  stdGram = std(GramVectorizedMat, [], 2);
  GramVectorizedMat = bsxfun(@rdivide, GramVectorizedMat, stdGram);

  [U, S, ~] = svd(GramVectorizedMat, 'econ');

  principal = U(:, 1) .* stdGram + meanGram;
  EigenGramLayers{layerI} = reshape(principal, sz(1), sz(2));
  
end


%generate white noise image;
imsz = net.meta.normalization.imageSize(1:3);
im0 = single(generateWhiteNoiseImage(imsz));
%generated image
im0_ = bsxfun(@minus,single(im0),avgImg) ;
%apply network on layer
imNew = vl_simplenn(net, gpuArray(im0_));

disp('generating new image');

Niterations = 500;
%step decreases by annealFactor every so often
annealFactor = gpuArray(0.5);

zerosGpu = zeros(size(imNew(1).x), 'gpuArray');


%record error every [plotInterval] timesteps
prevError = 0;
plotInterval = 5;
plotIndices = plotInterval:plotInterval:Niterations;
err = zeros(length(plotIndices), 1);
plotI = 1;

% [style content size variation]
gradWeights = gpuArray([0.0005 1 0.05 0.1]);
gradWeights = gradWeights ./ sum(gradWeights);

tolerance = 1e-5;

% l-bfgs parameters
step = gpuArray(0.1); %gradient des step size
[h,w,d] = size(imNew(1).x);
m =gpuArray(5);
s = zeros(h*w*d,m,'gpuArray');
y = zeros(h*w*d,m,'gpuArray');
alpha = zeros(1,m,'gpuArray');
beta = zeros(1,m,'gpuArray');
rou = zeros(1,m,'gpuArray');
gradPrev = zeros(h*w*d,1,'gpuArray');

for iter = 1:Niterations

    
    %gradient for style:
    gradStyle = zerosGpu;
    style_error = 0;

    if(useGradientStyleCombination)
        for sImage = 1 : numStyleImages
            [gradStyle_current, style_error_current] = computeGradStyle(net, imNew, ...
                GramStyleList(sImage).GramLayers, desiredLayers, desiredLayerWeights) ;
            gradStyle = gradStyle + gradStyle_current;
            style_error = style_error_current;
        end

        gradStyle = gradStyle ./ numStyleImages;
         style_error = style_error / numStyleImages;
    else
        [gradStyle, style_error] = computeGradStyle(net, imNew, ...
            EigenGramLayers, desiredLayers, desiredLayerWeights) ;
    end

    %gradient for Content
    diffContent = imNew(L+1).x - imContentNetL;
    gradNext = diffContent;
    gradNext(imNew(L+1).x < 0) = 0;
    %back prop with our functions
    gradContent = backProp(net, L, imNew, gradNext);    


%The following regularization errors are described in
%"Understanding Deep Image Representations by Inverting Them " 
%by Mahendran et. al, Univ. Oxford

    %error to limit size of values (avoid too large pixel values)
    %L(x(i,j)) = 0.5*x(i,j)^2 
    gradSize = bsxfun(@minus, imNew(1).x, imContentMean);

    %error to limit variation between pixels (like low pass filter)
    %L(x(i,j)) = 0.5( (x(i,j+1)-x(i,j))^2 + (x(i+1,j)-x(i,j))^2 )
    shiftRight = zerosGpu;
    shiftDown = shiftRight;
    shiftRight(:,1:end-1,:) = imNew(1).x(:,1:end-1,:)-imNew(1).x(:,2:end,:);
    shiftDown(1:end-1,:,:) = imNew(1).x(1:end-1,:,:)-imNew(1).x(2:end,:,:);
    gradVariation = shiftRight + shiftDown;

    grad = gradWeights(1)*gradStyle + gradWeights(2)*gradContent +  ...
      gradWeights(3)*gradSize + gradWeights(4)*gradVariation;


% ============== l-bfgs update =======================================================   
    if(iter > 2)
        gradPrev = grad1d;
    end %if
   
    grad1d = mtx2vec(grad);
    imNew1D = mtx2vec(imNew(1).x);
    imPrev = imNew1D;
    
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
        gradUpdate = vec2mtx(z,h,w,d);
        imNew(1).x = imNew(1).x - step*gradUpdate;
        s(:,1:m-1) = s(:,2:m);
        s(:,m) = mtx2vec(imNew(1).x) - imPrev;
    end %if
% =======================================================================

    %reapply image on network
    imNew = vl_simplenn(net, imNew(1).x);

    % record error if desired
    if iter == plotIndices(plotI) 
      errorsI = zeros(4, 1, 'gpuArray');
      errorsI(1) = style_error; 
      errorsI(2) = 0.5*sumsqr(diffContent);
      errorsI(3) = 0.5*sumsqr(gradSize);
      errorsI(4) = 0.5*(sumsqr(shiftRight) + sumsqr(shiftDown));%variation error

      errTotalI = gradWeights * errorsI; 

      err(plotI) = gather(errTotalI);
      disp(sprintf('iteration %03d, error: %.2f', iter, errTotalI));
      if plotI < length(plotIndices)
        plotI = plotI + 1;
      end

      %anneal step
      errDif  = prevError - errTotalI;
      if iter > 50 && (errDif < 0)
        step = annealFactor*step
        if step < tolerance
            disp('step is too small');
            break;
        end
      end

      prevError = errTotalI;

    end
end % for each iteration

%im2uint8 scales images, so use uint8
img = uint8(imNew(1).x);
img = gather(img);
save('img1.mat','img');

plotter;


