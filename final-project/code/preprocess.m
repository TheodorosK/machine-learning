clear all;
close all;
clc;

'Reading data...'
# Will read file using comma and space as delimiters
train = dlmread('../data/training.csv', ', ', 1, 0);
'Done.'

# Each row is an image with 96x96 pixels all laid out in a single vector
train_img = train(:, 31:end);

train_img_str = zeros(size(train_img));
for r=1:10 #rows(train_img)
	r
	train_img_str(r, :) = histo_stretch(train_img(r, :));
endfor

# try on one image
for r=1:10
	mimg = reshape(train_img(r, :), 96, 96)';
	mimg_str = reshape(train_img_str(r, :), 96, 96)';
	#figure(); imshow(mimg, [0, 255]); figure(); imshow(mimg_str, [0, 255]);
	subplot(1,2,1); imshow(mimg, [0, 255]);
	subplot(1,2,2); imshow(mimg_str, [0, 255]);
	pause()
endfor