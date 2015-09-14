
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREPARING TO GET ACCURACY ONLY ON THE TARGETED DATA - artificially moved data to test set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        MOTORBIKE --> BICYCLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% LOAD THE RESULT OF SPLME_CRP %%%%%%%%%%%%%%%%%%%%
load('/v9/exp_results/lmspe_crp_pascal3d_9640.mat'); % result{1 - 5}
load('/v9/pascal3d/exp_dataset/testset/features.mat'); % teF
load('/v9/pascal3d/exp_dataset/testset/labels.mat'); % teL


param = result{1};
W = result{2};
U = result{3};

%%%%%%%%%% DO GRAPH MATCHING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd /v9/code/VisualTransfer/pgm
addpath 'RRWM'
param_gm.maxIterGM = 10;
param_gm.match_thrsh = 0.1;
param_gm.match_sim_thrsh = 0.05;
param_gm.knn1 = 3;
param_gm.knn2 = 4;
param_gm.voting_alpha = 10;

U_bicycle = U(:, 11:16);
U_motorbike = U(:, 152:167);

[X_sol, cand_matches] = progGM(U_bicycle, U_motorbike, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);



load('/v9/pascal3d/exp_dataset/train_test_idx.mat') % tr_set_idx / te_set_idx
load('/v9/pascal3d/annot_mat/azimuth.mat') % azimuth_list
load('/v9/pascal3d/annot_mat/labels.mat') % labels

tr_labels = labels(tr_set_idx);
te_labels = labels(te_set_idx);
tr_azimuth = azimuth_list(tr_set_idx);
te_azimuth = azimuth_list(te_set_idx);

classNum = 2; % bicycle
classFrom = 120;
classTo = 150;
te_class_idx = find(te_labels == classNum);
te_class_azimuth = te_azimuth(te_class_idx);

class_azimuth_target = find(te_class_azimuth >= classFrom & te_class_azimuth < classTo);
class_idx_target = te_class_idx(class_azimuth_target); % target indices among te_labels


target = 16;
transferred_prototype = zeros(200, 1);
for n=1:numMatched
    transferred_prototype = transferred_prototype + U_bicycle(:, matched_pairs(n, 1)) - U_motorbike(:, matched_pairs(n, 2)) + U_motorbike(:, target);
end
transferred_prototype = transferred_prototype/numMatched;

U = [U(:, 1:16) transferred_prototype U(:, 17:end)];

% classIdx = find(teL == 2);
% class_feat = teF(:, classIdx);
class_feat = teF(:, class_idx_target);

[~, classified_raw]= max(class_feat'*W'*U, [], 2);
unique(classified_raw)
(numel(find(classified_raw == 12)) + numel(find(classified_raw == 17)))/numel(classified_raw)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% SUCCESS %%%%%%%%%%%%%%%
%        MOTORBIKE --> BICYCLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U = U_org;
target = 16;
transferred_prototype = zeros(200, 1);
for n=1:numMatched
    transferred_prototype = transferred_prototype + U_bicycle(:, matched_pairs(n, 1)) - U_motorbike(:, matched_pairs(n, 2)) + U_motorbike(:, target);
end
transferred_prototype = transferred_prototype/numMatched;

U = [U(:, 1:16) transferred_prototype U(:, 17:end)];

classIdx = find(teL == 2);
class_feat = teF(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U, [], 2);
unique(classified_raw)
numel(find(classified_raw == 12))/numel(classified_raw)
% (numel(find(classified_raw == 12)) + numel(find(classified_raw == 17)))/numel(classified_raw)



%%%%%%%%%%%%%%% SUCCESS %%%%%%%%%%%%%%%
%           BUS --> TRAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U_bus = U(:, 55:57);
U_train = U(:, 204:213);

[X_sol, cand_matches] = progGM(U_train, U_bus, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);

U = U_org;
target = 1;
transferred_prototype = zeros(200, 1);
for n=1:numMatched
    transferred_prototype = transferred_prototype + U_train(:, matched_pairs(n, 1)) - U_bus(:, matched_pairs(n, 2)) + U_bus(:, target);
end
transferred_prototype = transferred_prototype/numMatched;

U = [U(:, 1:213) transferred_prototype U(:, 214:end)];

classIdx = find(teL == 11);
class_feat = teF(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U, [], 2);
unique(classified_raw)
(numel(find(classified_raw == 204)) + numel(find(classified_raw == 214)))/numel(classified_raw)



%%%%%%%%%%%%%%% SUCCESS %%%%%%%%%%%%%%%
%            TRAIN --> BUS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U_bus = U(:, 55:57);
U_train = U(:, 204:213);

[X_sol, cand_matches] = progGM(U_train, U_bus, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);

not_matched = 1:10;
not_matched(unique(matched_pairs(:, 1))) = [];

U = U_org;
target = 1;
transferred_prototype = zeros(200, 1);
for n=1:numMatched
    transferred_prototype = transferred_prototype + U_bus(:, matched_pairs(n, 2)) - U_train(:, matched_pairs(n, 1)) + U_train(:, target);
end
transferred_prototype = transferred_prototype/numMatched;

U = [U(:, 1:57) transferred_prototype U(:, 58:end)];

classIdx = find(teL == 5);
class_feat = teF(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U, [], 2);
unique(classified_raw)
(numel(find(classified_raw == 55)) + numel(find(classified_raw == 58)))/numel(classified_raw)











