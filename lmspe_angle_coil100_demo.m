
cd /v9/code/VisualTransfer

addpath 'lme/lmspe_crp'
addpath 'ddcrp'
addpath 'util'

method = 'lmspe_crp';
dataset = 'coil100';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);


% Azimuth Clustering
classProtos = [];
numPrototypes = zeros(1, param.numClasses);

for c=1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));

    for n=1:6
        classProtos = [classProtos mean(X_c(:, (n-1)*8+1:n*8), 2)];
        numPrototypes(c) = numPrototypes(c) + 1;
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
highest_acc = 0.7;
iter_condition = 1;

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW_lmspe_crp(DS, W, U, param);
    U = learnU_lmspe_crp(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult('lmspe_crp_angles', param.dataset, accuracy, {param, W, U, classProtos, accuracy});

        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.00001;

    n = n + 1;
end




%------------------------------------------------------------------------
% Graph Transfer
%------------------------------------------------------------------------

fprintf('\n\n=============================================================\n');
fprintf('                    GRAPH TRANSFER\n');
fprintf('=============================================================\n\n');

fprintf('\nFinding class-pairs to transfer graph structures..\n');
graph_pairs = nchoosek(1:param.numClasses, 2);
scores = zeros(size(graph_pairs, 1), 1);
for n=1:size(graph_pairs, 1)
    A1 = param.knnGraphs{graph_pairs(n, 1)};
    A2 = param.knnGraphs{graph_pairs(n, 2)};

    L1 = laplacian(A1, 1);
    L2 = laplacian(A2, 1);
    scores(n) = prematching(L1, L2);
end
scoreMatrix = triu(ones(param.numClasses), 1);
scoreMatrix(~~scoreMatrix) = scores;

scoreMatrix = scoreMatrix + scoreMatrix';
[~, cand_class] = max(scoreMatrix, [], 2);

class_to_transfer = [];
for n=1:length(cand_class)
    if cand_class(cand_class(n)) ~= n
        class_to_transfer = [class_to_transfer; n cand_class(n)];
    end
end
fprintf('\n--------------------\n');
fprintf('Transfer start\n');
fprintf('--------------------\n');
fprintf(' Transfer List\n');
for n=1:size(class_to_transfer, 1)
    fprintf('%4d%4d\n', n, class_to_transfer(n));
end
fprintf('\n');

postfix = ceil(10000000*rand);
save(sprintf('/v9/exp_results/graph_trasnfer_list_%d.mat', postfix), 'scoreMatrix');
save(sprintf('/v9/exp_results/graph_trasnfer_list_%d.mat', postfix), 'class_to_transfer', '-append');
fprintf('graph transfer saved filename : %s\n', sprintf('/v9/exp_results/graph_trasnfer_list_%d.mat', postfix));

% for n=1:size(class_to_transfer, 1)
%     param_gm.maxIterGM = 10;
%     param_gm.match_thrsh = 0.1;
%     param_gm.match_sim_thrsh = 0.05;
%     param_gm.knn1 = 3;
%     param_gm.knn2 = 4;
%     param_gm.voting_alpha = 10;

%     U_bicycle = U(:, 11:16);
%     U_motorbike = U(:, 152:167);

%     [X_sol, cand_matches] = progGM(U_bicycle, U_motorbike, param_gm);
%     matched_pairs = cand_matches(find(X_sol), :);
%     numMatched = size(matched_pairs, 1);
% % end














