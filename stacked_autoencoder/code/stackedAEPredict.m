function [W1, W2, a1, a2, a3] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

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

a1 = data;
a2 = sigmoid(W1*a1 + repmat(b1,1,size(a1,2)));
a3 = sigmoid(W2*a2 + repmat(b2,1,size(a1,2)));

M = exp(softmaxTheta*a3);
M = bsxfun(@rdivide, M, sum(M));

[temp pred] = max(M);









% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
