function [XTrain, YTrain, XTest, YTest] = loadCIFARData(location)
    %% Load CIFAR Data from Specified Location
    % This function loads CIFAR data batches from a specified location and
    % combines them into training and test datasets.
    %
    % Inputs:
    %   location - The directory where CIFAR data batches are stored.
    %
    % Outputs:
    %   XTrain - Training images (4D array)
    %   YTrain - Training labels (categorical array)
    %   XTest - Test images (4D array)
    %   YTest - Test labels (categorical array)

    % Append 'data' subdirectory to the given location path
    location = fullfile(location, 'data');

    %% Loading Training Data
    % Load each CIFAR training batch using a helper function and concatenate them.
    [XTrain1, YTrain1] = loadBatchAsFourDimensionalArray(location, 'data_batch_1.mat');
    [XTrain2, YTrain2] = loadBatchAsFourDimensionalArray(location, 'data_batch_2.mat');
    [XTrain3, YTrain3] = loadBatchAsFourDimensionalArray(location, 'data_batch_3.mat');
    [XTrain4, YTrain4] = loadBatchAsFourDimensionalArray(location, 'data_batch_4.mat');
    [XTrain5, YTrain5] = loadBatchAsFourDimensionalArray(location, 'data_batch_5.mat');
    % Combine all training batches into single training datasets
    XTrain = cat(4, XTrain1, XTrain2, XTrain3, XTrain4, XTrain5);
    YTrain = [YTrain1; YTrain2; YTrain3; YTrain4; YTrain5];

    %% Loading Test Data
    % Load the CIFAR test batch.
    [XTest, YTest] = loadBatchAsFourDimensionalArray(location, 'test_batch.mat');
end

function [XBatch, YBatch] = loadBatchAsFourDimensionalArray(location, batchFileName)
    %% Load Individual CIFAR Batch File
    % This function loads a single CIFAR data batch file and restructures
    % the data into a format suitable for training/testing.
    %
    % Inputs:
    %   location - Directory where the batch file is stored.
    %   batchFileName - Name of the batch file to load.
    %
    % Outputs:
    %   XBatch - Images from the batch (4D array)
    %   YBatch - Labels from the batch (categorical array)

    % Load the batch file data
    s = load(fullfile(location, batchFileName));
    XBatch = s.data';
    XBatch = reshape(XBatch, 32, 32, 3, []); % Reshape into a 4D array
    XBatch = permute(XBatch, [2 1 3 4]); % Permute to match image format
    YBatch = convertLabelsToCategorical(location, s.labels); % Convert labels to categorical format
end

function categoricalLabels = convertLabelsToCategorical(location, integerLabels)
    %% Convert Integer Labels to Categorical Labels
    % This function converts integer labels to categorical labels using
    % the label names stored in the CIFAR dataset metadata.
    %
    % Inputs:
    %   location - Directory where the metadata file is stored.
    %   integerLabels - Array of integer labels.
    %
    % Outputs:
    %   categoricalLabels - Categorical array of labels.

    % Load label names from the metadata file
    s = load(fullfile(location, 'batches.meta.mat'));
    categoricalLabels = categorical(integerLabels, 0:9, s.label_names); % Convert to categorical
end
