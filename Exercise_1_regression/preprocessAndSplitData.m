function [X_train, Y_train, X_test, Y_test] = preprocessAndSplitData(data)
    % Split the data into training and testing sets
    cv = cvpartition(height(data), 'HoldOut', 0.3);
    
    % Specify features and target variable
    featureCols = {'GrLivArea', 'YearBuilt'};
    targetCol = 'SalePrice';
    
    % Extract features and target for training and testing
    X_train = data{training(cv), featureCols};
    Y_train = data{training(cv), targetCol};
    X_test = data{test(cv), featureCols};
    Y_test = data{test(cv), targetCol};
end
