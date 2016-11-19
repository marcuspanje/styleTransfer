%replace path with location of vl_setupnn from matlabconvnet package
%run('~/code/matconvnet-1.0-beta21/matlab/vl_setupnn');
run('C:\Users\Daniel Hsu\Documents\MATLAB\matconvnet-1.0-beta23\matlab\vl_setupnn');
%load weights of the trained vgg-face network
%this repo does not store the mat file. It can be obtianed from:
%http://www.robots.ox.ac.uk/~vgg/data/vgg_face/
loadNet = 0;
if loadNet
  net = load('vgg-face.mat');
  net = vl_simplenn_tidy(net);
end

%images must be 244x244
%im = imread('img/khan.jpg');
im = imread('C:\229\styleTransfer\img\khan.jpg');
%content
imC = applyNet(im, net);

% Show the classification result.
%{
scores = squeeze(gather(imC(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
              net.meta.classes.description{best}, best, bestScore), ...
      'Interpreter', 'none') ;
figure(2);
plot(scores);
%}
