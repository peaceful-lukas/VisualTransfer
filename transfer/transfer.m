function [U_new param_new U matched_pairs trainTargetClasses] = transfer(DS, W, U, c1, c2, scale_alpha, param)
% TRANSFER
%    transfer class prototypes ( c1 ---> c2 )
%    All the unmatched prototypes are transferred to the class c2


numProto_c1 = param.numPrototypes(c1);
numProto_c2 = param.numPrototypes(c2);
protoStartIdx = [0 cumsum(param.numPrototypes)];
U_c1 = U(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
U_c2 = U(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

simMatrix = U_c1'*U_c2;
sim_scores = sort(simMatrix(:), 'descend');

%%%%%% Graph Matching - RRWM
param_gm.maxIterGM = 10;
param_gm.match_thrsh = sim_scores(min(numProto_c1, numProto_c2));
param_gm.match_sim_thrsh = sim_scores(max(numProto_c1, numProto_c2));
param_gm.knn1 = 3;
param_gm.knn2 = 4;
param_gm.voting_alpha = 10;


[X_sol, cand_matches] = progGM(U_c1, U_c2, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);

% fprintf('    c1    c2  \n  ------------\n');
% disp(matched_pairs);



%%%%%% Transfer (c1 --> c2)
new_numPrototypes = param.numPrototypes;

unmatched = 1:param.numPrototypes(c1);
unmatched(matched_pairs(:, 1)) = [];

transferred_prototypes = [];
for um_idx=1:length(unmatched)
    target = unmatched(um_idx);
    transferred = zeros(param.lowDim, 1);
    for n=1:numMatched
        transferred = transferred + scale_alpha*U_c2(:, matched_pairs(n, 2)) - U_c1(:, matched_pairs(n, 1)) + U_c1(:, target);
    end
    transferred = transferred/numMatched;
    transferred_prototypes = [transferred_prototypes transferred];
end

[U_new param_new] = updatePrototypes(U, transferred_prototypes, c1, c2, matched_pairs, unmatched, param);
% U_new = [U(:, 1:sum(param.numPrototypes(1:c2))) transferred_prototypes U(:, sum(param.numPrototypes(1:c2))+1:end)];
% new_numPrototypes(c2) = new_numPrototypes(c2) + length(unmatched);


dispAccuracies(DS, W, U, U_new, new_numPrototypes, param);
trainTargetClasses = getClassesToBeLocallyTrained(DS, W, U, U_new, new_numPrototypes, param);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U_new param_new] = updatePrototypes(U, transferred_prototypes, c1, c2, matched, unmatched, param)
% 1. add transferred prototypes into U
% 2. update the number of prototypes of the class
% 3. update the knn-graph of the class

U_new = [U(:, 1:sum(param.numPrototypes(1:c2))) transferred_prototypes U(:, sum(param.numPrototypes(1:c2))+1:end)];

param_new = param;
param_new.numPrototypes(c2) = param.numPrototypes(c2) + length(unmatched);


A1 = param.knnGraphs{c1};
A2 = param.knnGraphs{c2};
A2_new = zeros(size(A2, 1)+length(unmatched));
A2_new(1:size(A2, 1), 1:size(A2, 1)) = A2;

n = size(A2, 1);
for i=1:length(matched)
    for j=1:length(unmatched)
        if A1(matched(i, 1), unmatched(j)) == 1
            A2_new(matched(i, 2), n+j) = 1;
        end
    end
end

A2_new(n+1:end, n+1:end) = A1(unmatched, unmatched);

param_new.knnGraphs{c2} = A2_new;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainTargetClasses = getClassesToBeLocallyTrained(DS, W, U, U_new, new_numPrototypes, param)
trainTargetClasses = [];
for cls = 1:param.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W, U, param);
    new_acc = getNewAccuracy(cls, DS, W, U_new, new_numPrototypes, param);
    if new_acc == orig_acc
        continue;
    else
        trainTargetClasses = [trainTargetClasses; cls];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function acc_list = dispAccuracies(DS, W, U, U_new, new_numPrototypes, param)
acc_list = [];
for cls = 1:param.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W, U, param);
    new_acc = getNewAccuracy(cls, DS, W, U_new, new_numPrototypes, param);
    acc_list = [acc_list; orig_acc new_acc];
    fprintf('Accuracy (class %d) : %.4f ----> %.4f\n', cls, orig_acc, new_acc);    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W, U, param)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
orig_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('ORIGINAL accuracy for class %d ----> %.4f\n', cls, orig_acc);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_acc = getNewAccuracy(cls, DS, W, U_new, new_numPrototypes, param)

cumNumProto = cumsum(new_numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U_new, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
new_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('TRANSFER accuracy for class %d ----> %.4f\n', cls, new_acc);




% cumNumProto = cumsum(param.numPrototypes);
% [~, classified_raw] = max(DS.T'*W'*U, [], 2);
% classified = zeros(numel(classified_raw), 1);
% for c = 1:param.numClasses
%     t = find(classified_raw <= cumNumProto(c));
%     classified(t) = c;
%     classified_raw(t) = Inf;
% end
% test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
% fprintf('ORIGINAL accuracy for all classes ----> %.4f\n', test_acc);


% cumNumProto = cumsum(new_numPrototypes);
% [~, classified_raw] = max(DS.T'*W'*U_new, [], 2);
% classified = zeros(numel(classified_raw), 1);
% for c = 1:param.numClasses
%     t = find(classified_raw <= cumNumProto(c));
%     classified(t) = c;
%     classified_raw(t) = Inf;
% end
% test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
% fprintf('TRANSFER accuracy for all classes ----> %.4f\n', test_acc);




