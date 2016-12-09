imNewDisp= (gather(bsxfun(@plus, imNew(1).x, avgImg)));
imContentDisp= (gather(bsxfun(@plus, imContentScaled, avgImg)));
imStyleDisp= (gather(bsxfun(@plus, imStyleScaled1, avgImg)));
gradWeights = gather(gradWeights);
desiredLayers = gather(desiredLayers);
%imwrite(imNewDisp, 'img.jpg');
save('data.mat', 'imNewDisp', 'imContentDisp', 'imStyleDisp', 'err', 'plotIndices', 'avgImg', ...
    'gradWeights', 'desiredLayers', 'contentImage', 'styleImageList');
