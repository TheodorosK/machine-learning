%% Housekeeping

close all;
clc;

%% Read data

disp('Reading data...')
train = dlmread('training.csv', ',', 1, 0);
disp('Done reading data.')

faces = train(:, 31:end); % each row is a face

%% Increase contrast with histogram stetching

disp('Stretching faces...')
for r = 1:size(faces, 1)
    if mod(r, 250) == 0
        fprintf('%.1f percent done\n', 100*r/size(faces, 1));
    end
    faces(r, :) = histogram_stretch(faces(r, :));
end
disp('Done stretching faces.')

%% Compute eigenfaces

avg_face = mean(faces);
imshow(reshape(avg_face, 96, 96)', [0 255]); % for fun only

faces = bsxfun(@minus, faces, mean(faces)); % remove average face

disp('Finding and ordering eigenvalues/eigenvectors...')
S = cov(faces);
[V, D] = eig(S);
eigval = diag(D);

eigval = eigval(end:-1:1); % re-order from highest to lowest
V = fliplr(V); % each column is a PC
disp('Done with eigenvalues/eigenvectors.')

%% Visualize some eigenfaces

figure;
for ii=1:6
    eigface = normalize_eigface(V(:, ii));
    subplot(2, 3, ii);
    imshow(reshape(eigface, 96, 96)', [0 255]);
    title(strcat('Eigenface',{' '},num2str(ii)));
    ax = gca;
    ax.FontSize = 8;
end
print('eigfaces', '-dpdf')

%% Scree plot and variance analysis

eigsum = sum(eigval);
csum = 0; %csum = zeros(size(faces,2));
tv = zeros(size(faces, 2),1);
for ii = 1:size(faces,2)
    csum = csum + eigval(ii);
    tv(ii) = csum / eigsum;
end;

% First 334 factors (out of 9216) explain 95 percent of variation
k95 = sum(tv<0.95);
k99 = sum(tv<0.99);

% Plot cumulative proportion of variance explained vs. how many PCs
figure;
plot(tv, '-k', 'linewidth', 2); hold on;
plot([k95 k95], [0 0.95], '-.r'); hold on;
plot([0 k95], [0.95 0.95], '-.r'); hold on;
plot([k99 k99], [0 0.99], '-.b'); hold on;
plot([0 k99], [0.99 0.99], '-.b'); hold on;
xlim([1 size(faces,2)]);
title('Principal Components Share of Variance Explained')
xlabel('Number of Principal Components'); 
ylabel('Cumulative Share of Variance Explained');
print('varplot', '-dpdf')

% Scree plot for first X eigenvalues
figure;
plot(eigval(1:50), '-k', 'linewidth', 2);
title('Scree Plot for Eigenfaces in Test Set');
xlabel('Principal Component');
ylabel('Eigenvalue');
print('screeplot', '-dpdf')
