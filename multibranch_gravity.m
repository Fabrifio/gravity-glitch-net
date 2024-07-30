% Code settings 
clear all
warning off

% Load dataset
datas = 1;
load(strcat('DatasGravity', int2str(datas)), 'DATA');

% Dataset order of patterns based on current fold (fold = row of indexes)
folderNumber = size(DATA{3}, 1);
datasetFolder = DATA{3};

% Prepare dataset for split between training and test set
trainSize = DATA{4};
totalSize = DATA{5};

% Retrieve all labels and patterns
y_true = DATA{2};
x_true = DATA{1};

% Load pre-trained AlexNet
%netAlex = alexnet;
%inputSize = [227 227];

% Transfer of AlexNet pretrained layers
%layersTransfer = netAlex.Layers(2 : 9);

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
inputSize = [210 250];

% Training parameters
miniBatchSize = 30;
learningRate = 1e-4;
optimizer = 'sgdm';
options = trainingOptions(optimizer,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(trainSize/miniBatchSize);

% For each folder
for fold = 1 : folderNumber
    close all force
    
    % Dataset train-test split on current fold
    trainPatternIndexes = (datasetFolder(fold, 1 : trainSize));
    testPatternIndexes = (datasetFolder(fold, trainSize + 1 : totalSize));
    y_fold = y_true(trainPatternIndexes);
    y_fold_test = y_true(testPatternIndexes);
    numClasses = max(y_fold);
    
    % Dataset train-validation split
    valSize = floor(trainSize * 0.1);
    

    % Create training set
    clear nome trainingImages
    for pattern = 1 : trainSize
        image = x_true{datasetFolder(fold, pattern)};
        
        % Rescale of image to a standard size for the CNN
        image = imresize(image, [inputSize(1) inputSize(2)]);
        if size(image, 3) == 1
            image(:, :, 2) = image;
            image(:, :, 3) = image(:, :, 1);
        end

        % Add image to training set
        trainingImages(:, :, :, pattern) = image;
    end


    % Get image size
    imageSize = size(image);
    
    % Data augmentation
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[1 2]);
    trainingImages = augmentedImageDatastore(imageSize, trainingImages, categorical(y_fold'), 'DataAugmentation', imageAugmenter);
    
    % Net tuning
    % The last three layers of the pretrained network net are configured for 1000 classes.
    % These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    %layersTransfer = net.Layers(1 : end-3);
    %layers = [
    %    layersTransfer
    %    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    %    softmaxLayer
    %    classificationLayer];
    netTransfer = trainNetwork(trainingImages, net, options);
    
    % Create test set
    clear nome test testImages
    for pattern = ceil(trainSize) + 1 : ceil(totalSize)
        image = x_true{datasetFolder(fold, pattern)};

        % Rescale of image to a standard size for the CNN
        image = imresize(image, [inputSize(1) inputSize(2)]);
        if size(image, 3) == 1
            image(:, :, 2) = image;
            image(:, :, 3) = image(:, :, 1);
        end

        % Add image to test set
        testImages(:, :, :, pattern - ceil(trainSize)) = uint8(image);
    end
    
    % Classifying test patterns
    [outclass, score{fold}] =  classify(netTransfer, testImages);
    
    % For each pattern, get highest confidence and related class
    [a, b] = max(score{fold}');

    % Get accuracy (correctly matched labels in test set divided by size)
    ACC(fold) = sum(b == y_fold_test) ./ length(y_fold_test);

    % Save whatever you need
    %%%%%
    
end


