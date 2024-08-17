% Code settings 
clear all
warning off

% Network input image size
inputSize = [280 340 3];

% Number of classes
numClasses = 22;

% Network structure
net = [
    % Input layer
    imageInputLayer(inputSize, 'Name', 'input')

    % Convolution layer 1
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1');
    reluLayer('Name', 'relu1');
    
    % Max-pooling 1
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1');
    % Dropout 1
    %dropoutLayer(0.5, 'Name', 'drop1');

    % Convolution layer 2
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv2');
    batchNormalizationLayer('Name', 'batchnorm2');
    reluLayer('Name', 'relu2');

    % Max-pooling 2
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2');
    % Dropout 2
    %dropoutLayer(0.5, 'Name', 'drop2');

    % Convolution layer 3
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv3');
    batchNormalizationLayer('Name', 'batchnorm3');
    reluLayer('Name', 'relu3');

    % Max-pooling 3
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3');
    % Dropout 3
    %dropoutLayer(0.5, 'Name', 'drop3');

    % Convolution layer 4
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv4');
    batchNormalizationLayer('Name', 'batchnorm4');
    reluLayer('Name', 'relu4');

    % Max-pooling 4
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4');
    % Dropout 4
    %dropoutLayer(0.5, 'Name', 'drop4');

    % Fully-connected layer 1
    fullyConnectedLayer(256, 'Name', 'fc1');
    batchNormalizationLayer('Name', 'batchnorm5');
    reluLayer('Name', 'relu5');
    % Dropout 5
    dropoutLayer(0.5, 'Name', 'drop5');

    % Fully-connected layer 2
    fullyConnectedLayer(numClasses, 'Name', 'fc2');
    softmaxLayer('Name', 'softmax');

    % Network output
    classificationLayer('Name', 'output');
];

% For each view dataset
for datas = 1 : 4
    close all force

    % Load dataset
    load(strcat('dataset/', 'DatasGravity', int2str(datas)), 'DATA');
    
    % Dataset order of patterns based on current fold (fold = row of indexes)
    % Load a fold of the first dataset for all datasets
    if(datas == 1)
        datasetFolder = DATA{3};
        fold = 1;
    end
    
    % Dataset sizes
    totalSize = DATA{5};
    trainValidationSize = DATA{4};
    
    % Number of instances per dataset
    trainSize = floor(trainValidationSize * 0.9);
    valSize = trainValidationSize - trainSize;
    testSize = totalSize - trainSize - valSize;
    
    % Retrieve all patterns and labels
    x_true = DATA{1};
    y_true = DATA{2};
    
    % TODO: unique shuffle (two ideas) to solve the classes issue
    % Dataset train-validation-test split on current fold
    trainPatternIndexes = datasetFolder(fold, 1 : trainSize);
    validationPatternIndexes = datasetFolder(fold, trainSize + 1 : trainValidationSize);
    testPatternIndexes = datasetFolder(fold, trainValidationSize + 1 : totalSize);
    
    % Label indexes to pick
    y_fold_train = y_true(trainPatternIndexes);
    y_fold_validation = y_true(validationPatternIndexes);
    y_fold_test = y_true(testPatternIndexes);
    
    % Create training set
    clear trainingImages
    for pattern = 1 : trainSize
        % Get image
        image = x_true{datasetFolder(fold, pattern)};
        % Rescale of image to a standard size for the network
        image = imresize(image, [inputSize(1) inputSize(2)]);
        % Add image to training set
        trainingImages(:, :, :, pattern) = uint8(image);
    end

    % TODO: evaluate addition
    % Data augmentation of training set
    % imageAugmenter = imageDataAugmenter('RandXReflection', true, 'RandXScale', [1 2]);
    % trainingImages = augmentedImageDatastore(inputSize, trainingImages, categorical(y_fold_train'), 'DataAugmentation', imageAugmenter);
    
    % Create validation set
    clear validationImages
    for pattern = trainSize + 1 : trainValidationSize
        % Get image
        image = x_true{datasetFolder(fold, pattern)};
        % Rescale of image to a standard size for the network
        image = imresize(image, [inputSize(1) inputSize(2)]);
        % Add image to validation set
        validationImages(:, :, :, pattern - trainSize) = uint8(image);
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
        'ValidationData', {validationImages, categorical(y_fold_validation')}, ...
        'ValidationFrequency', valFrequency, ...
        'OutputNetwork', 'best-validation', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

    % Network training without image augmentation
    netTransfer = trainNetwork(trainingImages, categorical(y_fold_train'), net, options);
    % TODO: evaluate replacement
    % Network training with image augmentation
    % netTransfer = trainNetwork(trainingImages, net, options);

    % Create test set
    clear testImages
    for pattern = trainValidationSize + 1 : totalSize
        % Get image
        image = x_true{datasetFolder(fold, pattern)};
        % Rescale of image to a standard size for the network
        image = imresize(image, [inputSize(1) inputSize(2)]);
        % Add image to test set
        testImages(:, :, :, pattern - trainValidationSize) = uint8(image);
    end
    
    % Classifying test patterns
    [outclass, score{fold}] =  classify(netTransfer, testImages);
    
    % Get highest confidence and related class for each pattern
    [a, b] = max(score{fold}');
    % Get accuracy (correctly matched labels in test set divided by size)
    acc(fold) = sum(b == y_fold_test) ./ length(y_fold_test);

    % Save trained and validated model
    save(strcat('models/gravity_d', int2str(datas), '_c4_f2.mat'), 'netTransfer');
end


