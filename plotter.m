imNewDisp= uint8(gather(bsxfun(@plus, imNew(1).x, avgImg)));
imwrite(imNewDisp, 'img.jpg');
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
