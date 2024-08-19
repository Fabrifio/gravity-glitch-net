% Code settings 
clear all
warning off

% Dataset to fill
DATASET = cell(1, 5);

% Create dataset
for datas = 1 : 1
    % Network load
    NET = load(strcat('models\gravity_d', int2str(datas), '_c4_f2'));
    
    % Dataset load
    load(strcat('dataset\DatasGravity', int2str(datas)), 'DATA')
    data_size = DATA{5};
    
    % Save dataset info
    if datas == 1
        for field = 2 : 5
            DATASET{field} = DATA{field};
        end
    end

    % Instances and labels
    x_true = DATA{1};
    y_true = DATA{2};

    % Input image size
    input_size = [280 340 3];

    % Classify instances
    for i = 1 : data_size
        % Rescale image and save features
        image = imresize(x_true{i}, [input_size(1) input_size(2)]);
        features(i, :) = activations(NET.netTransfer, image, 'fc1');
        DATASET{1}{i} = features(i, :);
    end

    % Save created dataset
    save(strcat('dataset/DatasGravityFeatures', int2str(datas)), 'DATASET');

    % Clear variables
    clear NET DATA features
end