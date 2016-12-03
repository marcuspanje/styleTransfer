imNewDisp= (gather(bsxfun(@plus, imNew(1).x, avgImg)));
imContentDisp= (gather(bsxfun(@plus, imContent(1).x, avgImg)));
imStyle = imStyles(1);
imStyleDisp= (gather(bsxfun(@plus, imStyle(1).x, avgImg)));
%imwrite(imNewDisp, 'img.jpg');
save('data.mat', 'imNewDisp', 'imContentDisp', 'imStyleDisp', 'err', 'plotIndices', 'avgImg');
