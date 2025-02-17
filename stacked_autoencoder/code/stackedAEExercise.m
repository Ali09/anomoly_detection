%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  stacked autoencoder exercise. 
%
%%======================================================================
%% STEP 0: Initializing parameters values

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.
addpath ../data/mnist
% Load MNIST database files
trainData = loadMNISTImages('train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels-idx1-ubyte');

num_images = 5000;
trainData = trainData(:,1:num_images);
trainLabels = trainLabels(1:num_images,:);

trainLabels(trainLabels == 0) = 10;

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%%
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

sae1OptTheta = sae1Theta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

addpath minFunc

[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  features.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% 
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta


sae2OptTheta = sae2Theta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);



% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%


lambda = 1e-4;
options.maxIter = 10;
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];
stackedAETheta = stackedAETheta;

%% checkStackedAECost

% checkStackedAECost();
%%
%{
stackedAEOptTheta = stackedAETheta; 

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

                          
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ...
                             numClasses, netconfig, lambda, trainData, trainLabels),...
                             stackedAETheta, options);



% -------------------------------------------------------------------------


%}
%}
%%======================================================================
%% STEP 6: Test 

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
%{
testData = loadMNISTImages('t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
%}

% filter trainData for specific numbers
num_filter = [3, 4, 5, 6];
new_trainData = [];
new_labels = [];
for i = 1:length(num_filter)
    new_trainData = [new_trainData, trainData(:,logical((trainLabels == num_filter(i))))];
    new_labels = [new_labels; num_filter(i) * ones(sum(trainLabels == num_filter(i)), 1)];
end

anom_filter = [2, 7];
anom_trainData = [];
anom_labels = [];
for i = 1:length(anom_filter)
    anom_trainData = [anom_trainData, trainData(:,logical((trainLabels == anom_filter(i))))];
    anom_labels = [anom_labels; anom_filter(i) * ones(sum(trainLabels == anom_filter(i)), 1)];
end


trainData = [trainData anom_trainData];

[W1, W2, a1, a2, a3] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, trainData);
csvwrite('ahidden1_train.txt', a2);
csvwrite('ahidden2_train.txt', a3);

for i = 1:length(trainData)
[~, train_error(i), ahidden1, ahidden2] = reconstructionError(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, trainData(:,i));
end

[maxError, maxIndexes] = sort(train_error, 'descend');
[minError, minIndexes] = sort(train_error, 'ascend');

sub_fig = figure;
for i = 1:20
   subplot(4,5,i) 
   imshow(reshape(trainData(:,maxIndexes(i)),28,28,1));
   title({['image ', num2str(maxIndexes(i))], ...
       ['rec error = ' num2str(maxError(i), 3)]});
end    
saveas(sub_fig, 'top_rec_error.png')

min_rand_samp = randsample(minIndexes(1:2500), 20);
sub_fig_2 = figure;
for i = 1:20
   subplot(4,5,i) 
   imshow(reshape(trainData(:,min_rand_samp(i)),28,28,1));
   title({['image ', num2str(min_rand_samp(i))]});
end    
saveas(sub_fig_2, 'min_rec_error.png')            

%{
acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
%}

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
