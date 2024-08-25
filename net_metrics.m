% Code settings
clear all
warning off

% Choose network
netChoice = input('Choose network:\n 0 = MLP (default)\n 1 = BiLSTM\n');

if netChoice == 0
    str = 'mlp';
elseif netChoice == 1
    str = 'bilstm';
elseif isempty(netChoice)
    netChoice = 0;
    str = 'mlp';
else
    error('Invalid choice');
end

% Load network
NET = load(strcat('models\gravity_', str, '.mat'));

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

% Get test set indexes and labels
testPatternIndexes = datasetFolder(fold, trainValidationSize + 1 : totalSize);
y_fold_test = y_true(testPatternIndexes);

% Create test set
clear testSequences;
for pattern = trainValidationSize + 1 : totalSize
    if netChoice == 0
        % Get sequence
        sequence = [x_true{1}{datasetFolder(fold, pattern)}'; 
            x_true{2}{datasetFolder(fold, pattern)}'; 
            x_true{3}{datasetFolder(fold, pattern)}';];

        % Add sequence to test set
        testSequences(pattern - trainValidationSize, :) = sequence;
    else
        % Get sequence
        sequence = [x_true{1}{datasetFolder(fold, pattern)}; 
            x_true{2}{datasetFolder(fold, pattern)}; 
            x_true{3}{datasetFolder(fold, pattern)}];

        % Transpose sequence
        sequence = sequence';
    
        % Add sequence to test set
        testSequences{pattern - trainValidationSize} = sequence;
    end
end

% Classifying test patterns
[outclass, score{fold}] =  classify(NET.netTransfer, testSequences);
 
% Get highest confidence and related class for each pattern
[a, b] = max(score{fold}');
% Get accuracy (correctly matched labels in test set divided by size)
acc(fold) = sum(b == y_fold_test) ./ length(y_fold_test);

% Compute confusion matrix
confMat = confusionmat(categorical(y_fold_test), categorical(outclass));

% Display confusion matrix
cm = confusionchart(confMat);

% Print test accuracy
fprintf('Test accuracy: %.4f\n', acc);