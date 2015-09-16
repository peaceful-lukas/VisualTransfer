cd /v9/code/VisualTransfer

addpath 'lme/lme_new'
addpath 'ddcrp'
addpath 'sc'
addpath 'util'
addpath 'pgm'
addpath 'pgm/RRWM'
addpath 'transfer'
addpath 'transfer/local_lme'

method = 'lme_new';
dataset = 'awa';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);

% % Spectral Clustering
% protoAssign = zeros(length(DS.DL), 1);
% classProtos = [];
% param.numPrototypes = zeros(1, param.numClasses);

% for c=1:param.numClasses

%     % select the data in class c
%     dataIdx_c = find(DS.DL == c);
%     X_c = DS.D(:, dataIdx_c);

%     % centralize and scale the data
%     X_c = X_c - repmat(mean(X_c, 2), 1, size(X_c, 2));
%     X_c = X_c/max(max(abs(X_c)));

%     % build affinity matrices
%     neighbor_num = 15;         %% Number of neighbors to consider in local scaling
%     scale = 0.04;
%     D_c = dist2(X_c', X_c');         %% Euclidean distance
%     A = exp(-D/(scale^2));       %% Standard affinity matrix (single scale)
    
%     %% Zero out diagonal
%     ZERO_DIAG = ~eye(size(X_c, 2));
%     A = A.*ZERO_DIAG;

%     %% spectral clustering
%     clusts_STD = gcut(A, 10);




%     [clusts_RLS, bestIdx, clu_score] = cluster_rotate(A_LS, CLUSTER_NUM_CHOICES);
%     clust_info = clusts_RLS{bestIdx};
%     numClusters = length(clust_info);
    
%     classProtos_c = zeros(size(X_c, 1), numClusters);
%     for n=1:numClusters
%         classProtos_c(:, n) = mean(X_c(:, clust_info{n}), 2);
%         protoAssign(dataIdx_c(clust_info{n})) = sum(param.numPrototypes(1:c-1)) + n;
%     end
%     classProtos = [classProtos classProtos_c];

%     param.numPrototypes(c) = numClusters;

%     fprintf('class %d clustering finished ( # of clusters = %d )\n', c, numClusters);
% end



% % ddCRP clustering
protoAssign = zeros(length(DS.DL), 1);
numPrototypes = zeros(1, param.numClasses);
classProtos = [];
for c = 1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    D = conDstMat(X_c);
    D = D./max(max(D));
    
    numData_c = size(X_c, 2);
    alpha = numData_c * 0.01;
    a = mean(mean(D));
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);
    numPrototypes(c) = numel(unique(ta));
    protoAssign(find(DS.DL == c)) = ta + sum(numPrototypes(1:c-1));

    % centroids of each cluster by examining ta
    for p = 1:numel(unique(ta))
        classProtos = [classProtos mean(X_c(:, find(ta == p)), 2)];
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


param.lowDim = sum(numPrototypes)-1;
[~, pca_score, ~] = pca(classProtos');
U = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U = normc(U);
W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');


n = 0;
highest_acc = 0.40;
highest_W = W;
highest_U = U;
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
        highest_W = W;
        highest_U = U;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;

    n = n + 1;
end

%------------------------------------------------------------------------

W = highest_W;
U = highest_U;

fprintf('[ transfer_demo.m ] is executed\n');
transfer_demo;


