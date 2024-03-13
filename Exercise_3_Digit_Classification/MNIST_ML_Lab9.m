%Copyright Jacob Vaught 2024

close all; clear; clc;

% Load data and initialize
load('mnist.mat');
testImages = reshape(test.images, [], numel(test.labels))';

% Training block
trainModel = false; % Change to true to train
if trainModel
    hiddenSizes = [300, 200, 100];
    net = patternnet(hiddenSizes, 'trainscg', 'crossentropy');
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotconfusion', 'plotroc'};
    net.performParam.regularization = 0.1;
    net.trainParam = struct('epochs', 1000, 'max_fail', 20, 'lr', 0.1);
    [net, ~] = train(net, reshape(training.images, [], numel(training.labels))', ind2vec(training.labels' + 1));
    save('trainedModel.mat', 'net');
else
    load('trainedModel.mat');
end

% Evaluation
testPredictions = net(testImages');
accuracy = mean(vec2ind(testPredictions) - 1 == test.labels');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Visualization
numImages = 5;
randomIndices = randperm(length(test.labels), numImages);
for i = 1:numImages
    subplot(2, ceil(numImages / 2), i);
    imshow(255 - rescale(test.images(:,:,randomIndices(i)), 0, 255), []);
    title(sprintf('Pred: %d | True: %d', vec2ind(testPredictions(:, randomIndices(i))) - 1, test.labels(randomIndices(i))));
end
