% Setup and data loading
clc; clear; close all;
[XTrain, TTrain, XValidation, TValidation] = loadCIFARData(pwd);

% Display random images from training set
figure; 
imshow(imtile(XTrain(:,:,:,randperm(size(XTrain,4),20)), 'ThumbnailSize', [96,96]));

% Data augmentation setup
augimdsTrain = augmentedImageDatastore([32 32 3], XTrain, TTrain, 'DataAugmentation', ...
    imageDataAugmenter('RandXReflection',true,'RandXTranslation',[-4 4],'RandYTranslation',[-4 4]), ...
    'OutputSizeMode', "randcrop");

% Define and analyze network
lgraph = resnetLayers([32 32 3], 10, 'InitialFilterSize', 3, 'InitialNumFilters', 16, ...
    'InitialStride', 1, 'InitialPoolingLayer', "none", 'StackDepth', [4 3 2], 'NumFilters', [16 32 64]);

analyzeNetwork(lgraph);
deepNetworkDesigner;

% Training options setup
options = trainingOptions("sgdm", 'InitialLearnRate', 0.1, 'MaxEpochs', 80, 'MiniBatchSize', 128, ...
    'VerboseFrequency', floor(size(XTrain,4)/128), 'Shuffle', "every-epoch", 'Plots', "training-progress", ...
    'Verbose', false, 'ValidationData', {XValidation,TValidation}, 'ValidationFrequency', floor(size(XTrain,4)/128), ...
    'LearnRateSchedule', "piecewise", 'LearnRateDropFactor', 0.1, 'LearnRateDropPeriod', 60);

% Conditional network training or loading
doTraining = false;
if doTraining
    net = trainNetwork(augimdsTrain, lgraph, options);
else
    load("trainedResidualNetwork.mat", "net");
end

% Evaluation and performance metrics
[YValPred, probs] = classify(net, XValidation);
disp("Training error: " + mean(classify(net, XTrain) ~= TTrain)*100 + "%")
disp("Validation error: " + mean(YValPred ~= TValidation)*100 + "%")

% Visualization of results and confusion matrix
figure; confusionchart(TValidation, YValPred);
figure; 
idx = randperm(size(XValidation,4),9);
for i = 1:9
    subplot(3,3,i); imshow(XValidation(:,:,:,idx(i)));
    title(char(YValPred(idx(i))) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
