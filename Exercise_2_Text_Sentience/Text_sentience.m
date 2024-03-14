%% Configuration and Data Loading
% Copyright Jacob Vaught, 2024
% Clearing workspace and closing all figures
clc; clear; close all;

% Setting up configuration
trainModel = false; % If true, the model will be trained; if false, a pre-trained model will be loaded
numTrees = 1000; % Number of trees for the Random Forest model

% Loading data from parquet files
testData = parquetread('test_data.parquet');
trainData = parquetread('train_data.parquet');

% Preparing training data
X_train = cell2mat(cellfun(@(x) x', trainData.embeddings, 'UniformOutput', false)); % Convert cell array to matrix
y_train = trainData.label; % Extracting labels for training data

% Preparing testing data
X_test = cell2mat(cellfun(@(x) x', testData.embeddings, 'UniformOutput', false)); % Convert cell array to matrix
y_test = testData.label; % Extracting labels for testing data
reviews = testData.text; % Extracting review texts for evaluation

%% Model Training or Loading
if trainModel
    % Setting up figure for Out-of-Bag (OOB) error visualization during training
    hFig = figure('Name', 'OOB Error During Training');
    xlabel('Number of Grown Trees'); ylabel('Out-of-Bag Classification Error'); hold on;
    
    % Training model with incremental number of trees and plotting OOB error
    for n = 1:numTrees
        tempModel = TreeBagger(n, X_train, y_train, 'Method', 'classification', 'OOBPrediction', 'On');
        plot(1:n, oobError(tempModel), 'b-', 'LineWidth', 2); drawnow;
    end
    hold off;
    save('trainedModel.mat', 'tempModel'); % Saving the trained model
else
    load('trainedModel.mat', 'tempModel'); % Loading a pre-trained model
end

%% Prediction and Performance Metrics
[y_pred, scores] = predict(tempModel, X_test); % Predicting labels for the test data
y_pred = str2double(y_pred); % Converting predicted labels from string to double
fprintf('Accuracy: %.2f%%\n', mean(y_pred == y_test) * 100); % Calculating and displaying accuracy

%% Feature Importance and Review Predictions
% This section is executed only if the model has been trained
if trainModel
    % Displaying feature importance scores
    figure; bar(tempModel.OOBPermutedVarDeltaError);
    xlabel('Feature Number'); ylabel('Feature Importance'); title('Feature Importance Scores');
end

% Displaying a set number of review sentiment predictions
numReviewsToShow = 5; fprintf('\nReview Sentiment Predictions:\n');
randIndices = randperm(length(reviews), numReviewsToShow); % Randomly selecting reviews to show
for i = randIndices
    % Printing out review and its predicted sentiment
    fprintf('Review %d: %s\nPredicted Sentiment: %d (Probability: %.2f%%), Correct: %d\n', ...
            i, reviews{i}, y_pred(i), max(scores(i,:)) * 100, y_test(i));
end
