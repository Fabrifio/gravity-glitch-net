% Code settings 
clear all
warning off

% Network time step size of input sequence
inputSize = 256;

% Number of classes
numClasses = 22;

% Network hyperparameters
numHiddenUnits = 128;

% Network structure
net = [
    % Input layer
    sequenceInputLayer(inputSize, 'Name', 'input')

    % BiLSTM layer
    bilstmLayer(numHiddenUnits, OutputMode="last", Name='bilstm')

    % Fully connected layer
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer

    % Classification layer
    classificationLayer
];

% Datasets patterns
x_true = cell(1, 4);

% Dataset common variables initialization
y_true = [];
datasetFolder = [];
fold = [];
totalSize = [];
trainValidationSize = [];

% Gather datasets
for datas = 1 : 4
    close all force
    
    % Load dataset
    load(strcat('dataset/DatasGravityFeatures', int2str(datas)), 'DATASET');

    % Store dataset patterns
    x_true{datas} = DATASET{1}; 
    
    % Save common dataset info
    if datas == 1
        % Get true labels
        y_true = DATASET{2};
        
        % Get fold
        datasetFolder = DATASET{3};
        fold = 1;
        
        % Dataset sizes
        totalSize = DATASET{5};
        trainValidationSize = DATASET{4};
    end

    % Clear used dataset
    clear DATASET
end

% Number of instances per dataset
trainSize = floor(trainValidationSize * 0.9);
valSize = trainValidationSize - trainSize;
testSize = totalSize - trainSize - valSize;

% Dataset train-validation-test split on current fold
trainPatternIndexes = datasetFolder(fold, 1 : trainSize);
validationPatternIndexes = datasetFolder(fold, trainSize + 1 : trainValidationSize);
testPatternIndexes = datasetFolder(fold, trainValidationSize + 1 : totalSize);
    
% Label indexes to pick
y_fold_train = y_true(trainPatternIndexes);
y_fold_validation = y_true(validationPatternIndexes);
y_fold_test = y_true(testPatternIndexes);

% Example of building a sentence
x = [x_true{1}{1}; x_true{2}{1}; x_true{3}{1}; x_true{4}{1}];
