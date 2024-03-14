%% Initialization and Data Loading
% Copyright Jacob Vaught, 2024
% Clearing workspace, closing all figures, and clearing command window
clc; close all; clear all;

% Reading dataset
filename = 'train.csv'; % The path to the dataset file
data = readtable(filename); % Reading the table from the CSV file

% Preprocessing data and splitting into training and testing sets
[X_train, Y_train, X_test, Y_test] = preprocessAndSplitData(data);

%% Initial Model Parameters
a = zeros(5,1); % Initialize coefficients for features and their interactions
b = 0; % Initialize intercept
learning_rates = [1e-15, 1e-15, 1e-15, 1e-15, 1e-14]; % Set learning rates for each parameter
iterations = 1000; % Number of iterations for gradient descent
plotIter = 10; % Iterations interval after which to update the plot
N = size(X_train, 1); % Total number of training examples

% Extracting individual features from training data for readability
Feature1 = X_train(:, 1); % First feature
Feature2 = X_train(:, 2); % Second feature

%% Visualization Setup
% Setting up the initial scatter plot for training data
fig = figure;
scatter3(Feature1, Feature2, Y_train, 'b'); % Original data scatter plot
hold on; % Keeping the scatter plot for further plotting
xlabel('Square Footage'); ylabel('Year Built'); zlabel('Y_train (Target)');
title('Progression of Model Training');
view(3); % Adjusting the view for better 3D visualization

%% Gradient Descent for Linear Regression
for iter = 1:iterations
    % Predicting values using current model parameters
    Y_pred = a(1) * Feature1 + a(3) * Feature1.^2 + a(2) * Feature2 + a(4) * Feature2.^2 + a(5) * (Feature1 .* Feature2) + b;
    errors = Y_train - Y_pred; % Calculating errors
    updates = [-2/N * Feature1 .* errors, -2/N * Feature2 .* errors, ...
               -2/N * (Feature1.^2) .* errors, -2/N * (Feature2.^2) .* errors, ...
               -2/N * (Feature1 .* Feature2) .* errors]; % Gradient for each parameter
    
    % Update model parameters using gradient descent (vectorized operation)
    a = a - learning_rates' .* sum(updates, 1)';
    b = b - 1e-14 * (-2/N) * sum(errors); % Update intercept

    % Update 3D surface plot every 'plotIter' iterations
    if mod(iter, plotIter) == 0
        % If the surface plot exists from a previous iteration, delete it before redrawing
        if exist('hSurf', 'var') && isvalid(hSurf)
            delete(hSurf);
        end
        % Creating a grid for 3D surface plot
        [gridX1, gridX2] = meshgrid(linspace(min(Feature1), max(Feature1), 20), ...
                                    linspace(min(Feature2), max(Feature2), 20));
        gridY = a(1) * gridX1 + a(2) * gridX2 + a(3) * gridX1.^2 + a(4) * gridX2.^2 + ...
                a(5) * (gridX1 .* gridX2) + b; % Evaluating the model on the grid
        hSurf = surf(gridX1, gridX2, gridY, 'EdgeColor', 'none', 'FaceAlpha', 0.5); % Drawing the surface
        title(sprintf('Iteration: %d', iter)); drawnow; % Updating title with iteration number
    end
end

hold off; % Release the figure for other plots
legend('Training Data', 'Regression Surface'); % Adding legend to the plot
