function [U_new param_new matched_pairs trainTargetClasses, score_GM] = transfer(DS, W_new, U_new, W0, U0, c1, c2, scale_alpha, param_new, param0)
% TRANSFER
%    transfer class prototypes ( c1 ---> c2 )
%    All the unmatched prototypes are transferred to the class c2


numProto_c1 = param0.numPrototypes(c1);
numProto_c2 = param0.numPrototypes(c2);
protoStartIdx = [0 cumsum(param0.numPrototypes)];
U_c1 = U0(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
U_c2 = U0(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

simMatrix = U_c1'*U_c2;
sim_scores = sort(simMatrix(:), 'descend');

%%%%%% Graph Matching - RRWM
param_gm.maxIterGM = 10;
param_gm.match_thrsh = sim_scores(min(numProto_c1, numProto_c2));
param_gm.match_sim_thrsh = sim_scores(max(numProto_c1, numProto_c2));
param_gm.knn1 = 3;
param_gm.knn2 = 4;
param_gm.voting_alpha = 10;


[X_sol, cand_matches, score_GM] = progGM(U_c1, U_c2, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);

% fprintf('    c1    c2  \n  ------------\n');
% disp(matched_pairs);



%%%%%% Transfer (c1 --> c2)
new_numPrototypes = param_new.numPrototypes;

unmatched = 1:param0.numPrototypes(c1);
unmatched(matched_pairs(:, 1)) = [];

if numel(unmatched) > 0
    fprintf('\n\nTransfer begins!!\n\n');
    transferred_prototypes = [];
    for um_idx=1:length(unmatched)
        target = unmatched(um_idx);
        transferred = zeros(param0.lowDim, 1);
        for n=1:numMatched
            transferred = transferred + U_c2(:, matched_pairs(n, 2)) - scale_alpha* (U_c1(:, matched_pairs(n, 1)) - U_c1(:, target));
        end
        transferred = transferred/numMatched;
        transferred_prototypes = [transferred_prototypes transferred];
    end

    [U_new param_new] = updatePrototypes(U_new, transferred_prototypes, c1, c2, matched_pairs, unmatched, param_new);
else
    fprintf('\n\nNo transfer...\n\n');
end

dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0);
trainTargetClasses = getClassesToBeLocallyTrained(DS, W_new, U_new, W0, U0, param_new, param0);

fprintf('Transfer Result : \n');
fprintf('BEFORE TRANSFER >>\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W0, U0, param0);
fprintf('AFTER TRANSFER >>\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W_new, U_new, param_new);
fprintf('\n\n');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U_new param_new] = updatePrototypes(U_new, transferred_prototypes, c1, c2, matched, unmatched, param_new)
% 1. add transferred prototypes into U
% 2. update the number of prototypes of the class
% 3. update the knn-graph of the class

U_new = [U_new(:, 1:sum(param_new.numPrototypes(1:c2))) transferred_prototypes U_new(:, sum(param_new.numPrototypes(1:c2))+1:end)];


param_new.numPrototypes(c2) = param_new.numPrototypes(c2) + length(unmatched);


A1 = param_new.knnGraphs{c1};
A2 = param_new.knnGraphs{c2};
A2_new = zeros(size(A2, 1)+length(unmatched));
A2_new(1:size(A2, 1), 1:size(A2, 1)) = A2;

n = size(A2, 1);
for i=1:size(matched, 1)
    for j=1:length(unmatched)
        if A1(matched(i, 1), unmatched(j)) == 1
            A2_new(matched(i, 2), n+j) = 1;
        end
    end
end

A2_new(n+1:end, n+1:end) = A1(unmatched, unmatched);

param_new.knnGraphs{c2} = A2_new;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainTargetClasses = getClassesToBeLocallyTrained(DS, W_new, U_new, W0, U0, param_new, param0)
trainTargetClasses = [];
for cls = 1:param_new.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W0, U0, param0);
    new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new);
    if new_acc == orig_acc
        continue;
    else
        trainTargetClasses = [trainTargetClasses; cls];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function acc_list = dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0)
acc_list = [];
for cls = 1:param_new.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W0, U0, param0);
    new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new);
    acc_list = [acc_list; orig_acc new_acc];
    fprintf('Accuracy (class %d) : %.4f ----> %.4f\n', cls, orig_acc, new_acc);    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W0, U0, param0)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param0.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W0'*U0, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param0.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
orig_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('ORIGINAL accuracy for class %d ----> %.4f\n', cls, orig_acc);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new)

cumNumProto = cumsum(param_new.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W_new'*U_new, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param_new.numClasses
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




