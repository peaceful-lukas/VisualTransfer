
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   1. K-NN graph should be a connected graph !!
%   2. fine-tuning
%   3. graph matching
%   4. transfer knowledge by computing the newly added prototype.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd /v9/code/VisualTransfer

addpath 'lme/lmspe_crp'
addpath 'ddcrp'
addpath 'util'

method = 'lmspe_crp';
dataset = 'pascal3d';

param = getParam(method, dataset);
param.lowDim = 100;

DS = loadDataset(param.dataset);

load('/v9/pascal3d/annot_mat/azimuth.mat'); % azimuth_list
load('/v9/pascal3d/annot_mat/labels.mat'); % labels
load('/v9/pascal3d/exp_dataset/train_test_idx.mat'); % tr_set_idx / te_set_idx


% Azimuth Clustering
tr_azimuth = azimuth_list(tr_set_idx);

classProtos = [];
numPrototypes = zeros(1, param.numClasses);

for c=1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    tr_azimuth_c = tr_azimuth(find(DS.DL == c));

    for azi=0:30:330
        azi_tg = find(tr_azimuth_c >= azi & tr_azimuth_c < azi+30);
        if azi_tg > 0
            classProtos = [classProtos mean(X_c(:, azi_tg), 2)];
            numPrototypes(c) = numPrototypes(c) + 1;
        end
    end
end



%------ should be connected graphs. MUST BE CHECKED. -----------
param.numPrototypes = numPrototypes;
[spTriplets knnGraphs] = generate_sp_triplets_by_crp(classProtos, param);
param.spTriplets = spTriplets;
param.knnGraphs = knnGraphs;


[~, pca_score, ~] = pca(classProtos');
U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U0 = normc(U0);
W0 = learnW_lmspe_crp(DS, W, U0, param); % initialize with pre-learned W.
% W = randn(param.lowDim, param.featureDim);
% W = W/norm(W, 'fro');


n = 0;
highest_acc = 0.5;
iter_condition = 1;
while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW_lmspe_crp(DS, W0, U, param);
    U = learnU_lmspe_crp(DS, W, U0, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult('lmspe_crp_azimuth', param.dataset, accuracy, {param, W, U, classProtos, accuracy});

        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.01;

    n = n + 1;
end

