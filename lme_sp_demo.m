
addpath 'lme/lme_sp'
addpath 'util'

method = 'lme_sp';
dataset = 'awa';

param = getParam(method, dataset);

DS = loadDataset(param.dataset);


% initialize prototypes by PCA with mean values of datasets for each class
U_feature = zeros(param.featureDim, param.numClasses);
for n=1:param.numClasses
    U_feature(:, n) = mean(DS.D(:, find(DS.DL == n)), 2);
end
[~, pca_score, ~] = pca(U_feature');
pca_score = [pca_score ones(param.numClasses, 1)];

W = randn(param.lowDim, param.featureDim);
U = pca_score(:, 1:param.lowDim)';
W = W/norm(W, 'fro');


n = 0;
highest_acc = 0;
while( n < param.maxAlter )
    fprintf('\n============================= Iteration %d =============================\n', n+1);
    W = learnW_lme_sp(DS, W, U, param);
    U = learnU_lme_sp(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, n+1, DS, W, U);
    
    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, C, accuracy});
        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    n = n + 1;
end


