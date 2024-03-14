%% Initialization and Data Loading
% Copyright Jacob Vaught, 2024
% Clearing figures, workspace, and command window
close all; clear; clc;

% Loading dataset
load('mnist.mat'); % MNIST dataset contains training and testing sets
testImages = reshape(test.images, [], numel(test.labels))'; % Reshaping test images for the network

%% Training Block
trainModel = false; % Set to true if you want to train the model, false to load an existing model
if trainModel
    % Configuration of the neural network
    hiddenSizes = [300, 200, 100]; % Setting the size of hidden layers
    net = patternnet(hiddenSizes, 'trainscg', 'crossentropy'); % Creating a pattern recognition network
    % Setting plotting functions for training
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotconfusion', 'plotroc'};
    net.performParam.regularization = 0.1; % Adding regularization to improve generalization
    net.trainParam = struct('epochs', 1000, 'max_fail', 20, 'lr', 0.1); % Setting training parameters
    
    % Training the network
    [net, ~] = train(net, reshape(training.images, [], numel(training.labels))', ...
                     ind2vec(training.labels' + 1)); % Training model with training images and labels
    save('trainedModel.mat', 'net'); % Saving the trained network
else
    load('trainedModel.mat'); % Loading an existing trained network
end

%% Evaluation
testPredictions = net(testImages'); % Predicting labels for test images
accuracy = mean(vec2ind(testPredictions) - 1 == test.labels'); % Calculating the accuracy of predictions
fprintf('Accuracy: %.2f%%\n', accuracy * 100); % Displaying the accuracy

%% Visualization
numImages = 5; % Number of images to display
randomIndices = randperm(length(test.labels), numImages); % Randomly picking indices of images
for i = 1:numImages
    subplot(2, ceil(numImages / 2), i); % Creating subplot for each image
    % Displaying image and its prediction versus actual label
    imshow(255 - rescale(test.images(:,:,randomIndices(i)), 0, 255), []); % Inverting colors for better visibility
    title(sprintf('Pred: %d | True: %d', vec2ind(testPredictions(:, randomIndices(i))) - 1, test.labels(randomIndices(i))));
end
