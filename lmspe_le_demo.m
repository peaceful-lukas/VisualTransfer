cd /home/twkim/code/VisualTransfer
addpath 'lme/lmspe_le'
addpath 'util'

method = 'lmspe_le';
dataset = 'pascal';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);

M = zeros(param.featureDim, param.numPrototypes*param.numClasses);
for c=1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    [~, cntrd] = kmeans(X_c', param.numPrototypes);
    M(:, (c-1)*param.numPrototypes+1:c*param.numPrototypes) = cntrd';

    fprintf('Clustering data of the %d-th class is done.\n', c);
end

param.spTriplets = generate_sp_triplets_by_kmeans(M, param, false);

W = randn(param.lowDim, param.featureDim);
U = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');
U = U/norm(U, 'fro');


n = 0;
while( n < param.maxAlter )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    W = learnW_lmspe_le(DS, W, U, M, param);
    U = learnU_lmspe_le(DS, W, U, M, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, M, accuracy});
        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    n = n + 1;
end
