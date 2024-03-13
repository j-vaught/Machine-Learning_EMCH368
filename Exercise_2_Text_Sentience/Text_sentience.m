%Copyright Jacob Vaught 2024

clc; clear; close all;

% Configuration and data loading
trainModel = false; % Set true to train model
numTrees = 1000;
testData = parquetread('test_data.parquet');
trainData = parquetread('train_data.parquet');
X_train = cell2mat(cellfun(@(x) x', trainData.embeddings, 'UniformOutput', false));
y_train = trainData.label;
X_test = cell2mat(cellfun(@(x) x', testData.embeddings, 'UniformOutput', false));
y_test = testData.label;
reviews = testData.text;

% Train model or load existing model
if trainModel
    hFig = figure('Name', 'OOB Error During Training');
    xlabel('Number of Grown Trees'); ylabel('Out-of-Bag Classification Error'); hold on;
    for n = 1:numTrees
        tempModel = TreeBagger(n, X_train, y_train, 'Method', 'classification', 'OOBPrediction', 'On');
        plot(1:n, oobError(tempModel), 'b-', 'LineWidth', 2); drawnow;
    end
    hold off; save('trainedModel.mat', 'tempModel');
else
    load('trainedModel.mat', 'tempModel');
end

% Prediction and performance metrics
[y_pred, scores] = predict(tempModel, X_test);
y_pred = str2double(y_pred);
fprintf('Accuracy: %.2f%%\n', mean(y_pred == y_test) * 100);

% Feature importance and review predictions if model was trained
if trainModel
    figure; bar(tempModel.OOBPermutedVarDeltaError);
    xlabel('Feature Number'); ylabel('Feature Importance'); title('Feature Importance Scores');
end

numReviewsToShow = 5; fprintf('\nReview Sentiment Predictions:\n');
randIndices = randperm(length(reviews), numReviewsToShow);
for i = randIndices
    fprintf('Review %d: %s\nPredicted Sentiment: %d (Probability: %.2f%%), Correct: %d\n', ...
            i, reviews{i}, y_pred(i), max(scores(i,:)) * 100, y_test(i));
end
