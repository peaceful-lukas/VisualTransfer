%  K-NN graph should be a connected graph !!

cd /v9/code/VisualTransfer

addpath 'lme/lmspe_crp'
addpath 'ddcrp'
addpath 'util'

method = 'lmspe_crp';
dataset = 'awa';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);

% ddCRP clustering
numPrototypes = zeros(1, param.numClasses);
classProtos = [];
for c = 1:param.numClasses
    %---------------------------- Distance based CRP
    X_c = DS.D(:, find(DS.DL == c));
    D = conDstMat(X_c);
    
    numData_c = size(X_c, 2);
    alpha = numData_c * 0.001;
    a = mean(mean(D));
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);
    numPrototypes(c) = numel(unique(ta));


    %---------------------------- Simiarity based CRP
    % X_c = DS.D(:, find(DS.DL == c));
    % S = conSimMat(X_c);
    % S = S/max(max(S));
    
    % numData_c = size(X_c, 2);
    % alpha = numData_c * 0.05;
    % a = 1;
    % [ta, ~] = ddcrp(S, 'lgstc', alpha, a);
    % numPrototypes(c) = numel(unique(ta));
    

    % centroids of each cluster by examining ta
    for p = 1:numel(unique(ta))
        classProtos = [classProtos mean(X_c(:, find(ta == p)), 2)];
    end

    fprintf('class %d clustering finished --> prototypes are set.\n', c);
end

%------ should be connected graphs. MUST BE CHECKED. -----------
param.numPrototypes = numPrototypes;
[spTriplets knnGraphs] = generate_sp_triplets_by_crp(classProtos, param);
param.spTriplets = spTriplets;
param.knnGraphs = knnGraphs;


[~, pca_score, ~] = pca(classProtos');
U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U0 = normc(U0);
W0 = randn(param.lowDim, param.featureDim);
W0 = W0/norm(W0, 'fro');
W0 = learnW_lmspe_crp(DS, W0, U0, param); % initialize with pre-learned W.


n = 0;
highest_acc = 0.6;
iter_condition = 1;

W = W0;
U = U0;

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW_lmspe_crp(DS, W0, U, param);
    U = learnU_lmspe_crp(DS, W, U0, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, classProtos, accuracy});

        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;

    n = n + 1;
end

