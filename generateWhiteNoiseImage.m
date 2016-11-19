function img =  generateWhiteNoiseImage(imgSize)
%imgSize (w, h, d): vector with 3 values denoting the 
%size of desired white noise image

img = zeros(imgSize);
for i = 1:imgSize(3)
  img(:, :, i) = round(255*rand(imgSize(1:2)));
end
img = uint8(img);
