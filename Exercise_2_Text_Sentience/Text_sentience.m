clc; clear; close all;

trainModel = false;
numTrees = 1000;
testData = parquetread('test_data.parquet');
trainData = parquetread('train_data.parquet');

X_train = cell2mat(cellfun(@(x) x', trainData.embeddings, 'UniformOutput', false));
y_train = trainData.label;

X_test = cell2mat(cellfun(@(x) x', testData.embeddings, 'UniformOutput', false));
y_test = testData.label;

reviews = testData.text; %load Reviews for later

if trainModel
    hFig = figure('Name', 'OOB Error During Training', 'NumberTitle', 'off');
    xlabel('Number of Grown Trees'); ylabel('Out-of-Bag Classification Error'); hold on;
    for n = 1:numTrees
        tempModel = TreeBagger(n, X_train, y_train, 'Method', 'classification',...
            'OOBPrediction', 'On', 'OOBPredictorImportance', 'on');
        oobErrorList = oobError(tempModel, 'Mode', 'ensemble');
        plot(1:n, oobErrorList, 'b-', 'LineWidth', 2); drawnow;
    end
    hold off; save('trainedModel.mat', 'tempModel', 'oobErrorList');
else
    load('trainedModel.mat', 'tempModel');
end

[y_pred, scores] = predict(tempModel, X_test);
y_pred = str2double(y_pred);
fprintf('Accuracy: %.2f%%\n', sum(y_pred == y_test) / numel(y_test) * 100);

if trainModel
    figure; bar(tempModel.OOBPermutedVarDeltaError);
    xlabel('Feature Number'); ylabel('Out-of-Bag Feature Importance'); title('Feature Importance Scores');
end

numReviewsToShow = 5; fprintf('\nReview Sentiment Predictions:\n 0: Negative\n 1: Positive');
randIndices = randperm(length(reviews), numReviewsToShow);
for i = randIndices
    fprintf('\nReview %d: %s\nPredicted Sentiment: %d (Probability: %.2f%%)\nCorrect Sentiment: %d\n', ...
            i, reviews{i}, y_pred(i), max(scores(i,:)) * 100, y_test(i));
end
