
addpath 'lme/lmspe'
addpath 'util'

param.dataset = 'awa';
param.numClasses = 200;
param.numPrototypes = 5;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 25;
param.batchSize = 10; % mini-batch size
param.lowDim = 200;
param.featureDim = 9216;

param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 1; % large margin for classification
param.sp_lm = 0.1; % large margin for structure preserving
param.lambda_W = 1000000; % regularizer coefficient
param.lambda_U = 10000; % regularizer coefficient
param.alpha = 5; % softmax parameter.
param.lr_W = 0.0001; % learning rate for W
param.lr_U = 0.0001; % learning rate for U
param.bal_c = 1;
param.bal_sp = 1;


%---- BEST RESULT(67.93%) for AwA dataset
% param.dataset = 'awa';
% param.numClasses = 50;
% param.numPrototypes = 10;
% param.maxIterW = 1000;
% param.maxIterU = 1000;
% param.maxAlter = 50;
% param.batchSize = 10; % mini-batch size
% param.lowDim = 100;
% param.featureDim = 9216;

% param.knn_const = 3; % constant for constructing k-nn graph.
% param.c_lm = 10; % large margin for classification
% param.sp_lm = 0.01; % large margin for structure preserving
% param.lambda_W = 1000; % regularizer coefficient
% param.lambda_U = 1000; % regularizer coefficient
% param.alpha = 5; % softmax parameter.
% param.lr_W = 0.00005; % learning rate for W
% param.lr_U = 0.00005; % learning rate for U
% param.bal_c = 1;
% param.bal_sp = 30;


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


    [~, classified] = max(DS.D'*W'*U, [], 2);
    classified = ceil(classified/param.numPrototypes);
    accuracy = numel(find(DS.DL == classified))/numel(DS.DL);
    fprintf('Alternation %d) train set accuracy : %.4f\n', n+1, accuracy);

    [~, classified] = max(DS.T'*W'*U, [], 2);
    classified = ceil(classified/param.numPrototypes);
    accuracy = numel(find(DS.TL == classified))/numel(DS.TL);
    fprintf('Alternation %d) TEST set accuracy :  %.4f\n', n+1, accuracy);
        
    if accuracy > highest_acc
        save_fname = sprintf('/home/twkim/exp_results/lmspe_%s_%.4f%%.mat', param.dataset, accuracy);
				save(save_fname, 'param');
				save(save_fname, 'W', '-append');
				save(save_fname, 'U', '-append');
				save(save_fname, 'C', '-append');
				save(save_fname, 'accuracy', '-append');

        highest_acc = accuracy;
				fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
		end

 


    n = n + 1;
end





