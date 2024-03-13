close all; clear; clc;
load('mnist.mat'); % This command loads the MNIST dataset from the 'mnist.mat' file.

testImages = reshape(test.images, [], numel(test.labels))';

if (false) %Set true if you WANT to train your model.
    trainingImages = reshape(training.images, [], numel(training.labels))';
    trainingLabels = training.labels;
    
    numClasses = 10; % There are 10 different digits(0-9)
    trainingLabels_dec = full(ind2vec(trainingLabels' + 1, numClasses));
    
    % Set Up the Neural Network:
    hiddenSizes = [300, 200, 100]; % This is the size of the hidden layers.
    net = patternnet(hiddenSizes); % This creates the neural network.
    
    % Set the training options.
    net.trainFcn = 'trainscg'; % This is the training function.
    net.performFcn = 'crossentropy'; % This is the performance function.
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', 'plotconfusion', 'plotroc'}; 
    net.performParam.regularization = 0.1; % This sets the regularization to reduce overfitting.
    net.trainParam.epochs = 1000; % This is the maximum number of training rounds.
    net.trainParam.max_fail = 20; % This is the maximum number of failures allowed during training.
    net.trainParam.lr = 0.1; % This is the learning rate.
    
    % Use the training images and their one-hot encoded labels to train the neural network.
    [net, tr] = train(net, trainingImages', trainingLabels_dec);
    save('trainedModel.mat', 'net');
end

load("trainedModel.mat")
% Use the trained network to predict the categories of the test images.
testPredictions = net(testImages');
testIndices = (vec2ind(testPredictions)-1)';
correctLabels = test.labels;
% Calculate and display the accuracy of the predictions.
accuracy = sum(testIndices == correctLabels) / numel(correctLabels);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

numImages = 5; % Number of random images to display
randomIndices = randperm(length(test.labels), numImages); % Randomly select indices
figure;
height_subplot = 2; % Number of rows in subplot grid
width_subplot = ceil(numImages / height_subplot); % Calculate number of columns based on desired height

for i = 1:numImages
    % Display each image
    subplot(height_subplot, width_subplot, i); % Positioning the image in the figure
    id = randomIndices(i); % Get the ID for the current random image
    invertedImage = 255 - rescale(test.images(:,:,id), 0, 255); % Subtracting from 255 inverts the image
    imshow(invertedImage, []); % Use imshow for correct scaling and display
    axis square; % Make sure the image is not distorted
    colormap(gray); % Use a grayscale colormap
    colorbar
    title(sprintf('Pred: %d | True: %d', testIndices(id), test.labels(id)), 'Interpreter', 'none');
end

