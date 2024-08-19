% Code settings 
clear all
warning off

% Network load
for datas = 1 : 4
    load(strcat('models\gravity_d', int2str(datas), '_c4_f2.mat'))
end

% load(strcat('dataset\DatasGravity', int2str(datas)))
% 
for datas = 1 : 4
    % Load
    load(strcat('dataset\DatasGravity', int2str(datas)))

    % Classify
    % for instance
       % get class and save

    % Save

    % Clear previous dataset
end

