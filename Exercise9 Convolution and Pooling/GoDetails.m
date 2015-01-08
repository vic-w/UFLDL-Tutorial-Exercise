clc; clear all; close all;

imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension
numPatches = 50000;    % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 

epsilon = 0.1;	       % epsilon for ZCA whitening

poolDim = 8;          % dimension of pooling region



load STL10Features.mat;

W1 = reshape(optTheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(optTheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = optTheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

patches = (W1*ZCAWhite)';

figure('Name', 'Patches');
displayColorNetwork( patches );


load stlTrainSubset.mat

images = reshape(trainImages, 64*64*3, 2000);
figure('Name', 'Train Images');
displayColorNetwork(images(:, 1:64));


load stlTestSubset.mat
images = reshape(testImages, 64*64*3, 3200);
figure('Name', 'Test Images');
displayColorNetwork(images(:, 1:64));


image = reshape(images(:,1), 64,64,3);
figure('Name', 'The First Test Image');
displayColorNetwork(images(:, 1));

convolvedFeatures = cnnConvolve(patchDim, hiddenSize, image, W1, b1, ZCAWhite, meanPatch);
conv = reshape( permute(convolvedFeatures, [3 4 2 1]), 57*57, 400);
figure('Name', 'Convolved Result');
display_network(conv);
    
pooledFeatures = cnnPool(poolDim, convolvedFeatures);
pool = reshape( permute(pooledFeatures, [3 4 2 1]), 7*7, 400);
figure('Name', 'PooledResult');
display_network(pool);

%reconstruct

rec_image = zeros(7*patchDim, 7*patchDim, imageChannels);

rec_patch = W2 * pool';
figure('Name', 'Reconstructed Patches');
displayColorNetwork(rec_patch);











