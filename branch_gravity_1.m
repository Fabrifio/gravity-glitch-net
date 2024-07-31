% Code settings 
clear all
warning off

% Load dataset
datas = 1;
load(strcat('DatasGravity', int2str(datas)), 'DATA');

% Dataset order of patterns based on current fold (fold = row of indexes)
datasetFolder = DATA{3};
folderNumber = size(datasetFolder, 1);

% Prepare dataset for split between training and test set
totalSize = DATA{5};
trainValidationSize = DATA{4};
% Number of instances per dataset
trainSize = floor(trainValidationSize * 0.9);
valSize = trainValidationSize - trainSize;
testSize = totalSize - trainSize - valSize;

% Retrieve all patterns and labels
x_true = DATA{1};
y_true = DATA{2};

% Load pre-trained AlexNet
% netAlex = alexnet;
% inputSize = [227 227];

% Transfer of AlexNet pretrained layers
% layersTransfer = netAlex.Layers(2 : 9);

% Load pre-trained AlexNet
net = [
    % Input layer
    imageInputLayer([210 250 3], 'Name', 'input')
    
    % Pretrained transfer layers from AlexNet
    %layersTransfer

    % Convolution layer 1
    convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm5');
    reluLayer('Name', 'relu1');
    
    % Max-pooling 1
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1');

    % Convolution layer 2
    convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv2');
    batchNormalizationLayer('Name', 'batchnorm5');
    reluLayer('Name', 'relu2');

    % Max-pooling 2
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2');

    % Convolution layer 3
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv3');
    batchNormalizationLayer('Name', 'batchnorm5');
    reluLayer('Name', 'relu3');

    % Max-pooling 3
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3');

    % Fully-connected layer 1
    fullyConnectedLayer(256, 'Name', 'fc1');
    batchNormalizationLayer('Name', 'batchnorm5');
    reluLayer('Name', 'relu5');
    dropoutLayer(0.5, 'Name', 'drop1');

    % Fully-connected layer 2 (Output layer)
    fullyConnectedLayer(22, 'Name', 'fc2');
    softmaxLayer('Name', 'softmax');
    classificationLayer('Name', 'output');
];

% Network input image size
% NOTE: different size with respect to paper for a faster training for testing
inputSize = [210 250 3];

% For each folder
for fold = 1 : folderNumber
    close all force
    
    % Dataset train-validation-test split on current fold
    % REMOVED: parenthesis
    trainPatternIndexes = datasetFolder(fold, 1 : trainSize);
    validationPatternIndexes = datasetFolder(fold, trainSize + 1 : trainValidationSize);
    testPatternIndexes = datasetFolder(fold, trainValidationSize + 1 : totalSize);
    % Label indexes to pick
    y_fold_train = y_true(trainPatternIndexes);
    y_fold_validation = y_true(validationPatternIndexes);
    y_fold_test = y_true(testPatternIndexes);
    
    % Number of possible classes
    % CHANGE: from max(y_fold_train)
    numClasses = 22;
    
    % Create training set
    % ADDITION: uint8 conversione to check
    clear trainingImages
    for pattern = 1 : trainSize
        image = x_true{datasetFolder(fold, pattern)};
        
        % Rescale of image to a standard size for the CNN
        image = imresize(image, [inputSize(1) inputSize(2)]);

        % Add image to training set
        trainingImages(:, :, :, pattern) = uint8(image);
    end

    % Get image size
    imageSize = inputSize;

    % Data augmentation
    imageAugmenter = imageDataAugmenter(...
        'RandXReflection', true, ...
        'RandXScale', [1 2]);
    trainingImages = augmentedImageDatastore(imageSize, trainingImages, categorical(y_fold_train'), 'DataAugmentation', imageAugmenter);
    
    % Net tuning
    % The last three layers of the pretrained network net are configured for 1000 classes.
    % These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    % layersTransfer = net.Layers(1 : end-3);
    % layers = [
    %    layersTransfer
    %    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    %    softmaxLayer
    %    classificationLayer];
    
    % Create validation set
    clear validationImages
    for pattern = trainSize + 1 : trainValidationSize
        image = x_true{datasetFolder(fold, pattern)};

        % Rescale of image to a standard size for the CNN
        image = imresize(image, [inputSize(1) inputSize(2)]);

        % Add image to test set
        validationImages(:, :, :, pattern - trainSize) = uint8(image);
    end

    % Training parameters
    miniBatchSize = 30;
    maxEpochs = 30;
    learningRate = 1e-4;
    optimizer = 'sgdm';
    valFrequency = 50;
    options = trainingOptions(optimizer, ...
        'MiniBatchSize', miniBatchSize, ...
        'MaxEpochs', maxEpochs, ...
        'InitialLearnRate', learningRate, ...
        'ValidationData', {validationImages, categorical(y_fold_validation')}, ...
        'ValidationFrequency', valFrequency, ...
        'OutputNetwork', 'best-validation', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

    % NOTE: useless?
    % numIterationsPerEpoch = floor(trainSize/miniBatchSize);

    % Network training
    netTransfer = trainNetwork(trainingImages, net, options);

    % Create test set
    clear testImages
    for pattern = trainValidationSize + 1 : totalSize
        image = x_true{datasetFolder(fold, pattern)};

        % Rescale of image to a standard size for the CNN
        image = imresize(image, [inputSize(1) inputSize(2)]);

        % Add image to test set
        testImages(:, :, :, pattern - trainValidationSize) = uint8(image);
    end
    
    % Classifying test patterns
    [outclass, score{fold}] =  classify(netTransfer, testImages);
    
    % For each pattern, get highest confidence and related class
    [a, b] = max(score{fold}');

    % Get accuracy (correctly matched labels in test set divided by size)
    ACC(fold) = sum(b == y_fold_test) ./ length(y_fold_test);

    % Save whatever you need
    save('gravity_c3_f2.mat', 'netTransfer');
end


