addpath 'lme/lmspe_crp'
addpath 'lme/lmspe_crp/spe'
addpath 'lme/lmspe_crp/spe/csdp'
addpath 'ddcrp'
addpath 'util'


param.numClasses = 20;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 40;
param.batchSize = 50; % mini-batch size
param.lowDim = 100;
param.featureDim = 4096;

param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 0.3; % large margin for classification
param.sp_lm = 0.01; % large margin for structure preserving
param.lambda_W = 10; % regularizer coefficient
param.lambda_U = 1000; % regularizer coefficient
param.alpha = 5; % softmax parameter.
param.lr_W = 0.0001; % learning rate for W
param.lr_U = 0.00001; % learning rate for U
param.bal_c = 1;
param.bal_sp = 30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Accuracy: 80.24%
% 
%   1. K-NN graph should be a connected graph !!
%   2. fine-tuning
%   3. graph matching
%   4. transfer knowledge by computing the newly added prototype.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% DS = loadDataset;

% ddCRP clustering
numPrototypes = zeros(1, param.numClasses);
classProtos = [];
for c = 1:param.numClasses
    %---------------------------- Distance based CRP
    % X_c = DS.D(:, find(DS.DL == c));
    % D = conDstMat(X_c);
    
    % numData_c = size(X_c, 2);
    % alpha = numData_c * 0.01;
    % a = mean(mean(D));
    % [ta, ~] = ddcrp(D, 'lgstc', alpha, a);
    % numPrototypes(c) = numel(unique(ta));


    %---------------------------- Simiarity based CRP
    X_c = DS.D(:, find(DS.DL == c));
    S = conSimMat(X_c);
    S = S/max(max(S));
    
    numData_c = size(X_c, 2);
    alpha = numData_c * 0.05;
    a = 1;
    [ta, ~] = ddcrp(S, 'lgstc', alpha, a);
    numPrototypes(c) = numel(unique(ta));



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
U = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U = normc(U);
W = randn(param.lowDim, param.featureDim);
W = W/norm(W, 'fro');


n = 0;
while( n < param.maxAlter )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    W = learnW_lmspe_crp(DS, W, U, param);
    U = learnU_lmspe_crp(DS, W, U, param);


    cumNumProto = cumsum(numPrototypes);
    [~, classified_raw] = max(DS.D'*W'*U, [], 2);
    classified = zeros(numel(classified_raw), 1);
    for c = 1:param.numClasses
        t = find(classified_raw <= cumNumProto(c));
        classified(t) = c;
        classified_raw(t) = Inf;
    end
    accuracy = numel(find(DS.DL == classified))/numel(DS.DL);
    fprintf('Alternation %d) train set accuracy : %.4f\n', n+1, accuracy);


    cumNumProto = cumsum(numPrototypes);
    [~, classified_raw] = max(DS.T'*W'*U, [], 2);
    classified = zeros(numel(classified_raw), 1);
    for c = 1:param.numClasses
        t = find(classified_raw <= cumNumProto(c));
        classified(t) = c;
        classified_raw(t) = Inf;
    end
    accuracy = numel(find(DS.TL == classified))/numel(DS.TL);
    fprintf('Alternation %d) TEST set accuracy :  %.4f\n', n+1, accuracy);

    n = n + 1;
end

