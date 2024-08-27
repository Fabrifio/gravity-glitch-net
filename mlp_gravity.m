% Code settings
clear all
warning off

% Set seed for better experimental reproducibility
rng(42);

% Network time step size of input sequence
inputSize = 768;

% Number of classes
numClasses = 22;

% Network hyperparameters
numHiddenUnits = 256;

% Network structure
net = [
    % Input layer
    featureInputLayer(inputSize, 'Name', 'input')

    % Fully connected layer 1
    fullyConnectedLayer(numHiddenUnits, 'Name', 'fc1')
    reluLayer('Name', 'relu')

    % Dropout 1
    dropoutLayer(0.5, 'Name', 'drop')

    % Fully connected layer 2
    fullyConnectedLayer(numClasses, 'Name', 'fc2')
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
    if datas ~= 1
        x_true{datas - 1} = DATASET{1}; 
    end
    
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

% Create training set
clear trainingSequences
for pattern = 1 : trainSize
    % Get sequence
    sequence = [x_true{1}{datasetFolder(fold, pattern)}'; 
        x_true{2}{datasetFolder(fold, pattern)}'; 
        x_true{3}{datasetFolder(fold, pattern)}';];

    % Add sequence to training set
    trainingSequences(pattern, :) = sequence;
end

% Create validation set
clear validationSequences
for pattern = trainSize + 1 : trainValidationSize
    % Get sequence
    sequence = [x_true{1}{datasetFolder(fold, pattern)}'; 
        x_true{2}{datasetFolder(fold, pattern)}'; 
        x_true{3}{datasetFolder(fold, pattern)}';];

    % Add sequence to validation set
    validationSequences(pattern - trainSize, :) = sequence;
end

% Training parameters
miniBatchSize = 30;
maxEpochs = 30;
learningRate = 1e-3;
optimizer = 'sgdm';
valFrequency = 50;
options = trainingOptions(optimizer, ...
        'MiniBatchSize', miniBatchSize, ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', learningRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 15, ...
        'LearnRateDropFactor', 0.1, ...
        'ValidationData', {validationSequences, categorical(y_fold_validation')}, ...
        'ValidationFrequency', valFrequency, ...
        'OutputNetwork', 'last-iteration', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

% Network training without image augmentation
netTransfer = trainNetwork(trainingSequences, categorical(y_fold_train'), net, options);

% Create test set
clear testSequences;
for pattern = trainValidationSize + 1 : totalSize
    % Get sequence
    sequence = [x_true{1}{datasetFolder(fold, pattern)}'; 
        x_true{2}{datasetFolder(fold, pattern)}'; 
        x_true{3}{datasetFolder(fold, pattern)}';];

    % Add sequence to test set
    testSequences(pattern - trainValidationSize, :) = sequence;
end

% Classifying test patterns
[outclass, score{fold}] =  classify(netTransfer, testSequences);
 
% Get highest confidence and related class for each pattern
[a, b] = max(score{fold}');
% Get accuracy (correctly matched labels in test set divided by size)
acc(fold) = sum(b == y_fold_test) ./ length(y_fold_test);

% Save trained and validated model
save('models/gravity_mlp.mat', 'netTransfer');