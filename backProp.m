function [grad] = backProp(net, L, imNew, gradNext)
%L: final layer of CNN
%gradNext: derivative of error wrt to output of final layer 
%imNew: outputs of each layer
%grad: derivative of error wrt to input


    for layer = fliplr(1:L)
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
end
