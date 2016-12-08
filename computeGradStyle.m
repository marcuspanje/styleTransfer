function [gradStyle, style_error] = computeGradStyle(net, imNew, GramLayers, ...
 desiredLayers, desiredLayerWeights)
%compute the gradient of style error w.r.t input image
%INPUT
%net : pretrained neural network
%imNew: activations of output image (initialized to white noise) at each layer of network
%GramLayers: cell of the gram matrix at desired layers.    
%GramLayers{i} is gram matrix of desiredLayer(i)
%desiredLayers: vector of desired layers
%desiredLayerWeights: vector of desired layer weights
%OUTPUT
%gradStyle: gradient (H,W,D) matrix (same dim as input picture)
%style_error: scalar error 

    %gradient for style:
    % recompute gradNext ----------------------
    % equ(6) in 'Gatys_Image_Style_Transfer_CVPR_2016_paper'
    style_error = gpuArray(0);
    gradStyle = zeros(size(imNew(1).x), 'gpuArray');
    for layerI = 1:length(desiredLayers);
        l = desiredLayers(layerI);
        w_l = desiredLayerWeights(layerI);
        netI.layers = net.layers(1:l);
        [h0,w0,d0] = size(imNew(l+1).x);
        nParams = h0*w0*d0;
        F = to2D(imNew(l+1).x);
        G = Gram(F);
        A = GramLayers{layerI};
        diffStyle = G-A;
        gradNext = (1/nParams^2)*(F'*(diffStyle))';
        gradNext(F<0)=0;
        gradNext = single(toND(gradNext,h0,w0));
        %apply backward pass
        gradStyleLayerI = backProp(net, l, imNew, gradNext);    

        gradStyle = gradStyle + w_l*gradStyleLayerI; 
        style_error = style_error + w_l*sumsqr(diffStyle)/(4*nParams^2);
    end
end
