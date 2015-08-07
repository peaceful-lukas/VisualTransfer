function DS = loadDataset(dataset)
    
    tic;
    train_feat_dir = ['/home/twkim/' dataset '/Features/train/'];
    [D D_labels] = loadFeatureData(train_feat_dir);
    train_data_dir = ['/home/twkim/' dataset '/Resized/train/'];
%    DI = loadImageData(train_data_dir);
    toc
    
    tic
    test_feat_dir = ['/home/twkim/' dataset '/Features/test/'];
    [T T_labels] = loadFeatureData(test_feat_dir);
    test_data_dir = ['/home/twkim/' dataset '/Resized/test/'];
 %   TI = loadImageData(test_data_dir);
    toc

    DS = {};
		DS.dataset = dataset;
    DS.D = D;
    DS.DL = D_labels;
 %   DS.DI = DI;
    DS.T = T;
    DS.TL = T_labels;
 %   DS.TI = TI;

end



function [F L] = loadFeatureData(data_dir)

    data = dir([data_dir '*.mat']);
    label_idx = numel(data);
    
    % load labels.
    load([data_dir data(label_idx).name]); % variable(labels) loaded.
    L = labels;
    
    % load features.    
    data(label_idx) = [];
    num_data = numel(data);
    F = zeros(9216, num_data);

    
    for n=1:num_data
        % fprintf('%d / %d\n', n, num_data);
        load([data_dir data(n).name]);
        F(:, n) = feat';
    end
end

function I = loadImageData(data_dir)
    data = dir([data_dir '*.jpg']);

    I = {};
    for n=1:numel(data)
        I{n} = imread([data_dir data(n).name]);
    end
end
