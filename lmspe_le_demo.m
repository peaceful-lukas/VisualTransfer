
addpath 'lmspe_le'

param.numClasses = 20;
param.numPrototypes = 10;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 30;
param.batchSize = 50; % mini-batch size
param.lowDim = 100;
param.featureDim = 4096;


param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 0.1; % large margin for classification
param.sp_lm = 0.001; % large margin for structure preserving
param.lambda_W = 10; % regularizer coefficient
param.lambda_U = 100; % regularizer coefficient
param.alpha = 5; % softmax parameter.
param.lr_W = 0.0001; % learning rate for W
param.lr_U = 0.00001; % learning rate for U
param.bal_c = 1;
param.bal_sp = 1;


DS = loadDataset('pascal3d');

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


    [~, classified] = max(DS.D'*W'*U*M, [], 2);
    classified = ceil(classified/param.numPrototypes);
    accuracy = numel(find(DS.DL == classified))/numel(DS.DL);
    fprintf('Alternation %d) train set accuracy : %.4f\n', n+1, accuracy);

    [~, classified] = max(DS.T'*W'*U*M, [], 2);
    classified = ceil(classified/param.numPrototypes);
    accuracy = numel(find(DS.TL == classified))/numel(DS.TL);
    fprintf('Alternation %d) TEST set accuracy :  %.4f\n', n+1, accuracy);

    n = n + 1;
end


RES = [min(DS.T'*W'*U, [], 2) max(DS.T'*W'*U, [], 2)];


