%Copyright Jacob Vaught 2024

clc; close all; clear all

filename = 'train.csv';  % Replace with your actual file path
data = readtable(filename);
[X_train, Y_train, X_test, Y_test] = preprocessAndSplitData(data);

a = zeros(5,1);  % Coefficients for features and their interactions
b = 0;   % Intercept
learning_rates = [1e-15, 1e-15, 1e-15, 1e-15, 1e-14];  % Learning rates for parameters
iterations = 1000;
plotIter = 10;
N = size(X_train, 1);  % Number of training examples

Feature1 = X_train(:, 1);
Feature2 = X_train(:, 2);

fig = figure;
scatter3(Feature1, Feature2, Y_train, 'b');  % Original data scatter plot
hold on;
xlabel('Square Footage'); ylabel('Year Built'); zlabel('Y_train (Target)');
title('Progression of Model Training');
view(3);

% Gradient Descent
for iter = 1:iterations
    Y_pred = a(1) * Feature1 + a(3) * Feature1.^2 + a(2) * Feature2 + a(4) * Feature2.^2 + a(5) * (Feature1 .* Feature2) + b;
    errors = Y_train - Y_pred;
    updates = [-2/N * Feature1 .* errors, -2/N * Feature2 .* errors, -2/N * (Feature1.^2) .* errors, -2/N * (Feature2.^2) .* errors, -2/N * (Feature1 .* Feature2) .* errors];
    
    % Update model parameters (vectorized)
    a = a - learning_rates' .* sum(updates, 1)';
    b = b - 1e-14 * (-2/N) * sum(errors);

    if mod(iter, plotIter) == 0
        if exist('hSurf', 'var') && isvalid(hSurf)
            delete(hSurf);
        end
        [gridX1, gridX2] = meshgrid(linspace(min(Feature1), max(Feature1), 20), linspace(min(Feature2), max(Feature2), 20));
        gridY = a(1) * gridX1 + a(2) * gridX2 + a(3) * gridX1.^2 + a(4) * gridX2.^2 + a(5) * (gridX1 .* gridX2) + b;
        hSurf = surf(gridX1, gridX2, gridY, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        title(sprintf('Iteration: %d', iter)); drawnow; 
    end
end

hold off;
legend('Training Data', 'Regression Surface');
