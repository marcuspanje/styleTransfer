inputFace;
%generate white noise image;
imsz = net.meta.normalization.imageSize;
im0 = generateWhiteNoiseImage(imsz);
%generated image
imR =  applyNet(im0, net); 

%Code outline for gradient descent across each layer. 
%Please edit the outline as implementation changes.
%im0 should be updated until it matches the content of the image
%above applied to the trained network

%entering gradient descent

%reference layer
figure(1);
L = 1;
gradNext = imR(L+1).x - imC(L+1).x;
step = 0.01;
Niterations = 10;
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
      szW = size(weights);
      pad = net.layers{layer}.pad;
      padYprev = zeros(szYprev + 2*pad);
      padYprev(pad+1:pad+szYprev(1), pad+1:pad+szYprev(2), ...
        pad+1:pad+szYprev(3)) = imR(layer).x; 
      
      for i = 1:szYprev(1)
        for j = 1:szYprev(2)
          for k = 1:szYprev(3)

            for filter = 1:szW(4)
              for a = 0:szW(1)-1
                for b = 0:szW(2)-1
                  for c = 0:szW(3)-1
                    %if out of bounds, continue
                    p = i-a; q = j-b; r = k-c;
                    if p <= 0 || q <= 0 || r<= 0
                      continue;
                    end
                    grad(i,j,k) = grad(i,j,k) + ...
                      gradNext(p,q,r) * weights(a+1,b+1,c+1,filter); 
                  end
                end
              end
            end
          end
        end
      end

    elseif strcmp(type, 'softmax')
      %gradient = softmaxGD(params) * gradient;
    elseif strcmp(type, 'relu')
      %gradient = reluGD(params) * gradient;
      
        for i = 1:szYprev(1)
            for j = 1:szYprev(2)
                for k = 1:szYprev(3)
                    if(gradNext(i, j, k) < 0)
                        grad(i, j, k) = 1;
                    else
                        grad(i, j, k) = 0;
                    end
                end
            end
        end
      
    elseif strcmp(type, 'pool')
      %gradient = reluGD(pool) * graident;
    end 
    gradNext = grad;

  end %for each layer

  imR(1).x = imR(1).x - step*grad;
  %reapply network on image
  imR = vl_simplenn(net, imR(1).x);
end % for each iteration

figure(2);
subplot(121);
imshow(imC(1).x);
title('reference');
subplot(122);
imshow(imR(1).x);
title('generated');
