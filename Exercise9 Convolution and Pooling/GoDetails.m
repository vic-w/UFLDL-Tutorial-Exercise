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

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

patches = (W*ZCAWhite)';

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

convolvedFeatures = cnnConvolve(patchDim, hiddenSize, image, W, b, ZCAWhite, meanPatch);
conv = reshape( permute(convolvedFeatures, [3 4 2 1]), 57*57, 400);
figure('Name', 'Convolved Result');
display_network(conv);
    
pooledFeatures = cnnPool(poolDim, convolvedFeatures);
pool = reshape( permute(pooledFeatures, [3 4 2 1]), 7*7, 400);
figure('Name', 'PooledResult');
display_network(pool);












