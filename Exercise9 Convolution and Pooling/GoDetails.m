clc; clear all; close all;

imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension
numPatches = 50000;    % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 

epsilon = 0.1;	       % epsilon for ZCA whitening

poolDim = 19;          % dimension of pooling region



load STL10Features.mat;

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

patches = (W*ZCAWhite)';

figure(1);
displayColorNetwork( patches );


load stlTrainSubset.mat
images = reshape(trainImages, 64*64*3, 2000);
figure(2);
displayColorNetwork(images(:, 1:64));

image = reshape(images(:,1), 64,64,3);

tic();

convolvedFeatures = cnnConvolve(patchDim, hiddenSize, image, W, b, ZCAWhite, meanPatch);
    
pooledFeatures = cnnPool(poolDim, convolvedFeatures);

toc();











