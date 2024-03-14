function [X_train, Y_train, X_test, Y_test] = preprocessAndSplitData(data)
    %% Preprocess and Split Data
    % This function preprocesses the housing dataset and splits it into training and testing sets.
    %
    % Inputs:
    %   data - A table containing the dataset with features and the target variable.
    %
    % Outputs:
    %   X_train - Training set features.
    %   Y_train - Training set target variable.
    %   X_test - Test set features.
    %   Y_test - Test set target variable.

    % Split the data into training and testing sets using a holdout method
    % Here, 70% of data is used for training and 30% for testing.
    cv = cvpartition(height(data), 'HoldOut', 0.3);
    
    % Specify the names of the feature columns and the target variable column
    featureCols = {'GrLivArea', 'YearBuilt'}; % Example features: 'GrLivArea' (living area square feet), 'YearBuilt' (year of construction)
    targetCol = 'SalePrice'; % Target variable: 'SalePrice' (price of house)
    
    % Extract features and target variable for the training set
    % 'training(cv)' returns logical indices for the training set rows
    X_train = data{training(cv), featureCols};
    Y_train = data{training(cv), targetCol};
    
    % Extract features and target variable for the test set
    % 'test(cv)' returns logical indices for the test set rows
    X_test = data{test(cv), featureCols};
    Y_test = data{test(cv), targetCol};
end
