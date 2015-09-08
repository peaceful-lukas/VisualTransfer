cd /v9/code/VisualTransfer

addpath 'lme/lme_new'
addpath 'ddcrp'
addpath 'util'
addpath 'pgm'
addpath 'pgm/RRWM'

method = 'lme_new';
dataset = 'pascal3d_all';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);


%% Azimuth Clustering
protoAssign = zeros(length(DS.DL), 1);
numPrototypes = zeros(1, param.numClasses);
classProtos = [];
for c = 1:param.numClasses
    class_idx = find(DS.DL == c);
    X_c = DS.D(:, class_idx);
    tr_azimuth_c = DS.DA(class_idx);
    
    for azi=1:12
        azi_idx = find(tr_azimuth_c == azi);
        if length(azi_idx) > 0
            classProtos = [classProtos mean(X_c(:, azi_idx), 2)];
            numPrototypes(c) = numPrototypes(c) + 1;
            protoAssign(class_idx(azi_idx)) = sum(numPrototypes);
        end
    end

    fprintf('class %d clustering finished ( # of clusters = %d )\n', c, numPrototypes(c));
end

% ------ should be connected graphs. MUST BE CHECKED. -----------
param.numPrototypes = numPrototypes;
param.cTriplets = generateClassificationTriplets(DS, param);
param.pTriplets = generateClusterPullingTriplets(protoAssign, param.numPrototypes);
[param.sTriplets knnGraphs] = generateStructurePreservingTriplets(classProtos, param);
param.knnGraphs = knnGraphs;
param.protoAssign = protoAssign;


[~, pca_score, ~] = pca(classProtos');
U = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U = normc(U);
W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');


n = 0;
highest_acc = 0.7;
iter_condition = 1;

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW_new(DS, W, U, param);
    U = learnU_new(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, classProtos, accuracy});

        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;

    n = n + 1;
end

