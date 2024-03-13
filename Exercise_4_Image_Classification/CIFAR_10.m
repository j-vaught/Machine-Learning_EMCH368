%Copyright Jacob Vaught 2024

clc; clear; close all;

% Load data and set up augmentation and network
[XTrain, TTrain, XValidation, TValidation] = loadCIFARData(pwd);
augOptions = imageDataAugmenter('RandXReflection',true,'RandXTranslation',[-4 4],'RandYTranslation',[-4 4]);
augimdsTrain = augmentedImageDatastore([32 32 3], XTrain, TTrain, 'DataAugmentation', augOptions, 'OutputSizeMode', "randcrop");
lgraph = resnetLayers([32 32 3], 10, 'InitialFilterSize', 3, 'InitialNumFilters', 16, ...
    'InitialStride', 1, 'InitialPoolingLayer', "none", 'StackDepth', [4 3 2], 'NumFilters', [16 32 64]);

% Set training options
options = trainingOptions("sgdm", 'InitialLearnRate', 0.1, 'MaxEpochs', 80, 'MiniBatchSize', 128, ...
    'Shuffle', "every-epoch", 'Plots', "training-progress", 'ValidationData', {XValidation,TValidation}, ...
    'LearnRateSchedule', "piecewise", 'LearnRateDropFactor', 0.1, 'LearnRateDropPeriod', 60);

% Train or load network
if ~exist('trainedResidualNetwork.mat', 'file')
    net = trainNetwork(augimdsTrain, lgraph, options);
else
    load("trainedResidualNetwork.mat", "net");
end

% Evaluation and display results
[YValPred, probs] = classify(net, XValidation);
disp("Training error: " + mean(classify(net, XTrain) ~= TTrain)*100 + "%")
disp("Validation error: " + mean(YValPred ~= TValidation)*100 + "%")
figure; confusionchart(TValidation, YValPred);
figure; idx = randperm(size(XValidation,4),9);
for i = 1:9
    subplot(3,3,i); imshow(XValidation(:,:,:,idx(i))); title(char( ...
        YValPred(idx(i))) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end