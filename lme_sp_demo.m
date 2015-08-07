
addpath 'lme_sp'

%------ BEST RESULT(80.xx%) for pascal VOC 2007 dataset
% param.numClasses = 20;
% param.lowDim = 100;
% param.featureDim = 4096;
% param.maxIterW = 1000;
% param.maxIterU = 1000;
% param.maxAlter = 20;
% param.batchSize = 100; % mini-batch size

% param.lr_W = 0.0001; % learning rate for W
% param.lr_U = 0.00001; % learning rate for U
% param.lm = 0.1; % large margin for classification
% param.lambda_W = 10; % regularizer coefficient
% param.lambda_U = 10; % regularizer coefficient

%------ BEST RESULT(69.xx%) for AwA dataset
% param.dataset = 'awa';
% param.numClasses = 50;
% param.lowDim = 50;
% param.featureDim = 9216;
% param.maxIterW = 1000;
% param.maxIterU = 1000;
% param.maxAlter = 50;
% param.batchSize = 30; % mini-batch size

% param.lr_W = 0.0001; % learning rate for W
% param.lr_U = 0.00001; % learning rate for U
% param.lm = 20; % large margin for classification
% param.lambda_W = 100; % regularizer coefficient
% param.lambda_U = 100; % regularizer coefficient





param.dataset = 'pascal3d';
param.numClasses = 12;
param.lowDim = 30;
param.featureDim = 9216;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 20;
param.batchSize = 50; % mini-batch size

param.lr_W = 0.0001; % learning rate for W
param.lr_U = 0.00001; % learning rate for U
param.lm = 10; % large margin for classification
param.lambda_W = 100; % regularizer coefficient
param.lambda_U = 100; % regularizer coefficient


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

    [~, classified] = max(DS.T'*W'*U, [], 2);
    accuracy = numel(find(DS.TL == classified))/numel(DS.TL);
    fprintf('Alternation %d) TEST data set accuracy : %.4f\n', n+1, accuracy);
    
    if accuracy > highest_acc
        save_fname = sprintf('/home/twkim/exp_results/lme_sp_%s_%.4f%%.mat', param.dataset, accuracy);
				save(save_fname, 'param');
				save(save_fname, 'W', '-append');
				save(save_fname, 'U', '-append');
				save(save_fname, 'accuracy', '-append');
				highest_acc = accuracy;
				fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
		end

    n = n + 1;
end


