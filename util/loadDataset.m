function DS = loadDataset(dataset)

    if strcmp(dataset, 'pascal3d_all')
        fprintf('PASCAL3D+ FULL DATA (manipulated dataset along with POSE ANNOTATION).\n\n');
        load('/v9/PASCAL3D/both/proc/trF.mat');
        load('/v9/PASCAL3D/both/proc/trL.mat');
        load('/v9/PASCAL3D/both/proc/trA.mat');
        load('/v9/PASCAL3D/both/proc/trB.mat');

        load('/v9/PASCAL3D/both/proc/teF.mat');
        load('/v9/PASCAL3D/both/proc/teL.mat');
        load('/v9/PASCAL3D/both/proc/teA.mat');
        load('/v9/PASCAL3D/both/proc/teB.mat');

        DS = {};
        DS.D = trF;
        DS.DL = trL;
        DS.DA = trA;
        DS.DB = trB;
        DS.T = teF;
        DS.TL = teL;
        DS.TA = teA;
        DS.TB = teB;
    elseif strcmp(dataset, 'pascal3d_imagenet')
        fprintf('WARNING) PASCAL3D+ DATA (manipulated dataset along with POSE ANNOTATION).\n\n');

        load('/v9/pascal3d/exp_dataset/trainset/features.mat');
        load('/v9/pascal3d/exp_dataset/trainset/labels.mat');
        % % load('/v9/pascal3d/exp_dataset/trainset/images.mat');

        load('/v9/pascal3d/exp_dataset/testset/features.mat');
        load('/v9/pascal3d/exp_dataset/testset/labels.mat');
        % load('/v9/pascal3d/exp_dataset/testset/images.mat');

        DS = {};
        DS.D = trF;
        DS.DL = trL;
        % DS.DI = trI;
        DS.T = teF;
        DS.TL = teL;
        %DS.TI = teI;
    elseif strcmp(dataset, '3dobj')
        load('/v9/3Ddataset/exp_dataset/trF.mat');
        load('/v9/3Ddataset/exp_dataset/teF.mat');
        load('/v9/3Ddataset/exp_dataset/trL.mat');
        load('/v9/3Ddataset/exp_dataset/teL.mat');

        DS = {};
        DS.D = trF;
        DS.DL = trL;
        DS.T = teF;
        DS.TL = teL;
    elseif strcmp(dataset, 'coil100')
        load('/v9/coil100/exp_dataset/trF.mat');
        load('/v9/coil100/exp_dataset/teF.mat');
        load('/v9/coil100/exp_dataset/trL.mat');
        load('/v9/coil100/exp_dataset/teL.mat');

        DS = {};
        DS.D = trF;
        DS.DL = trL;
        DS.T = teF;
        DS.TL = teL;
    else
        
        tic;
        train_feat_dir = ['/v9/' dataset '/Features/train/'];
        [D D_labels] = loadFeatureData(train_feat_dir);
        %train_data_dir = ['/v9/' dataset '/Resized/train/'];
        %DI = loadImageData(train_data_dir);
        toc

        tic
        test_feat_dir = ['/v9/' dataset '/Features/test/'];
        [T T_labels] = loadFeatureData(test_feat_dir);
        %test_data_dir = ['/v9/' dataset '/Resized/test/'];
        %TI = loadImageData(test_data_dir);
        toc

        DS = {};
        DS.D = D;
        DS.DL = D_labels;
        %DS.DI = DI;
        DS.T = T;
        DS.TL = T_labels;
        %DS.TI = TI;
    end

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
