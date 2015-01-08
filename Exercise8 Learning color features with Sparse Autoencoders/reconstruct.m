close all;

theta = optTheta;
data = patches(:, 1:100);

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

[ndims, m] = size(data);


z2 = zeros(hiddenSize, m);
z3 = zeros(visibleSize, m);
a2 = zeros(size(z2));
% a3 = zeros(size(z3));

z2 = bsxfun(@plus, W1*data, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2*a2, b2);


figure;
displayColorNetwork(data);

figure;
displayColorNetwork(z3);