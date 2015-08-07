
addpath 'lme/lmspe'
addpath 'util'

method = 'lmspe';
dataset = 'awa';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);


% K-means clustering
C = zeros(param.numPrototypes*param.numClasses, param.featureDim);
for c=1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    % [~, cntrd] = kmeans(X_c', param.numPrototypes, 'Display', 'iter');
    [~, cntrd] = kmeans(X_c', param.numPrototypes);
    C((c-1)*param.numPrototypes+1:c*param.numPrototypes, :) = cntrd;

    fprintf('Clustering data of the %d-th class is done.\n', c);
end
[~, pca_score, ~] = pca(C);

param.spTriplets = generate_sp_triplets_by_kmeans(C', param, false);


W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');
U = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U = normc(U);


n = 0;
highest_acc = 0;
while( n < param.maxAlter )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    W = learnW_lmspe(DS, W, U, param);
    U = learnU_lmspe(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, C, param);
    
        
    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, C, accuracy});
        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    n = n + 1;
end





