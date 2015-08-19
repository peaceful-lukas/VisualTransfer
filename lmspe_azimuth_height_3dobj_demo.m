
cd /v9/code/VisualTransfer

addpath 'lme/lmspe_crp'
addpath 'ddcrp'
addpath 'util'

method = 'lmspe_crp';
dataset = '3dobj';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);

load('/v9/3Ddataset/exp_dataset/teBlocks.mat');
load('/v9/3Ddataset/exp_dataset/trP.mat');

tr_pose = trP;
trBlocks = ~teBlocks;

% Azimuth Clustering
classProtos = [];
numPrototypes = zeros(1, param.numClasses);

for c=1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    tr_pose_c = tr_pose(find(DS.DL == c), :);
    trBlock = trBlocks(:, :, c);
    [azmth, hght]= find(trBlock);

    for n=1:16
        pose_idx = find(tr_pose_c(:, 1) == azmth(n) & tr_pose_c(:, 2) == hght(n));

        if length(pose_idx) > 0
            classProtos = [classProtos mean(X_c(:, pose_idx), 2)];
            numPrototypes(c) = numPrototypes(c) + 1;
        end
    end
    fprintf('class %d clustering done.\n', c);
end



%------ should be connected graphs. MUST BE CHECKED. -----------
param.numPrototypes = numPrototypes;
[spTriplets knnGraphs] = generate_sp_triplets_by_crp(classProtos, param);
param.spTriplets = spTriplets;
param.knnGraphs = knnGraphs;


[~, pca_score, ~] = pca(classProtos');
U = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U = normc(U);
W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');


n = 0;
highest_acc = 0.5;
iter_condition = 1;

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW_lmspe_crp(DS, W, U, param);
    U = learnU_lmspe_crp(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult('lmspe_crp_azimuth_height', param.dataset, accuracy, {param, W, U, classProtos, accuracy});

        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.00001;

    n = n + 1;
end

