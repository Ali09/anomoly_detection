function [output, error, ahidden] = reconstructionError(theta, visibleSize, hiddenSize, data)
 
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
% notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data. So, data(:,i) is the i-th training example. 
 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes.
 
% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

[sae1Theta, netconfig] = stack2params(stack(1));

[sae2Theta, netconfig] = stack2params(stack(2));

W1 = reshape(sae1Theta(1:hiddenSize*inputSize),hiddenSize, inputSize);
W2 = reshape(sae2Theta(1:hiddenSize*hiddenSize),hiddenSize, hiddenSize);

b1 = reshape(sae1Theta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize),hiddenSize, 1);
b2 = reshape(sae2Theta(hiddenSize*hiddenSize+1:hiddenSize*hiddenSize+hiddenSize),hiddenSize, 1);
 
%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
% and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc. Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1. I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j). Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
%
 
[nFeatures, nSamples] = size(data);
% first calculate the regular cost function J
 
[~, ahidden, output] = getActivation(W1, W2, b1, b2, data);
error = (sum((output(:) - data(:)) .^ 2)/sum(data(:).^2))^0.5;
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients. This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).
 
function sigm = sigmoid(x)
 
sigm = 1 ./ (1 + exp(-x));
end
 
%-------------------------------------------------------------------
% This function return the activation of each layer
%
function [ainput, ahidden, aoutput] = getActivation(W1, W2, b1, b2, input)
 
ainput = input;
ahidden = bsxfun(@plus, W1 * ainput, b1);
ahidden = sigmoid(ahidden);
aoutput = bsxfun(@plus, W2 * ahidden, b2);
aoutput = sigmoid(aoutput);
end