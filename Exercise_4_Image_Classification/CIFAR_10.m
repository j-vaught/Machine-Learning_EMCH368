%% Initial Setup
% Copyright Jacob Vaught, 2024
% Cleaning the workspace and closing all figures
clc; clear; close all;

%% Data Preparation and Augmentation
% Load CIFAR-10 data and set up data augmentation
[XTrain, TTrain, XValidation, TValidation] = loadCIFARData(pwd); % Loading data from current directory
augOptions = imageDataAugmenter('RandXReflection',true,'RandXTranslation',[-4 4],...
    'RandYTranslation',[-4 4]); % Defining data augmentation options
% Creating augmented image data store for training
augimdsTrain = augmentedImageDatastore([32 32 3], XTrain, TTrain, 'DataAugmentation', augOptions, ...
    'OutputSizeMode', "randcrop");

%% Network Configuration
% Setting up ResNet architecture customized for CIFAR-10
lgraph = resnetLayers([32 32 3], 10, 'InitialFilterSize', 3, 'InitialNumFilters', 16, ...
    'InitialStride', 1, 'InitialPoolingLayer', "none", 'StackDepth', [4 3 2], 'NumFilters', [16 32 64]);

%% Training Configuration
% Defining training options
options = trainingOptions("sgdm", 'InitialLearnRate', 0.1, 'MaxEpochs', 80, 'MiniBatchSize', 128, ...
    'Shuffle', "every-epoch", 'Plots', "training-progress", 'ValidationData', {XValidation,TValidation}, ...
    'LearnRateSchedule', "piecewise", 'LearnRateDropFactor', 0.1, 'LearnRateDropPeriod', 60);

%% Training Process
% Train the network if not already trained and saved
if ~exist('trainedResidualNetwork.mat', 'file')
    net = trainNetwork(augimdsTrain, lgraph, options); % Training the network
    save('trainedResidualNetwork.mat', 'net'); % Saving the trained network
else
    load("trainedResidualNetwork.mat", "net"); % Loading the trained network
end

%% Evaluation and Results Display
% Evaluating the network on validation data
[YValPred, probs] = classify(net, XValidation); % Classifying validation data
% Displaying training and validation errors
disp("Training error: " + mean(classify(net, XTrain) ~= TTrain)*100 + "%")
disp("Validation error: " + mean(YValPred ~= TValidation)*100 + "%")
% Confusion matrix for validation predictions
figure; confusionchart(TValidation, YValPred);
% Displaying random validation images with their predicted labels and confidence scores
figure; idx = randperm(size(XValidation,4),9);
for i = 1:9
    subplot(3,3,i); 
    imshow(XValidation(:,:,:,idx(i))); 
    title(char(YValPred(idx(i))) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
