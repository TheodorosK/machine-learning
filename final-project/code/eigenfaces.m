clear all;
close all;
clc;

% Will read file using comma and space as delimiters
train = dlmread('../data/training.csv', ', ', 1, 0);

faces = train(:, 31:end)'; % each column is a face
faces = bsxfun(@minus, faces, mean(faces))'; % de-meaned faces

S = cov(faces); % covariance matrix

[V,D] = eig(S); % eigenvectors/values
eigval = diag(D); 

eigval = eigval(end:-1:1); 
V = fliplr(V); 

% http://bytefish.de/pdf/facerec_octave.pdf
% https://en.wikipedia.org/wiki/Eigenface