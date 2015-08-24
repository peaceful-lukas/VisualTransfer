function [U_new U] = transfer(DS, W, U, c1, c2, target, param)

% target : the prototype in class 'c1'

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


%%%%%% Transfer (c1 --> c2)
transferred_prototype = zeros(param.lowDim, 1);
for n=1:numMatched
    transferred_prototype = transferred_prototype + U_c2(:, matched_pairs(n, 2)) - U_c1(:, matched_pairs(n, 1)) + U_c1(:, target);
end
transferred_prototype = transferred_prototype/numMatched;

U_new = [U(:, 1:target) transferred_prototype U(:, target+1:end)];


%%%%%% Disp Accuracy
cumNumProto = cumsum(param.numPrototypes);

classIdx = find(DS.TL == c2);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
orig_acc = numel(find(DS.TL == classified))/numel(DS.TL);
fprintf('ORIGINAL accuracy for class %d: %f\n', c2, orig_acc);



classIdx = find(DS.TL == c2);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U_new, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
new_acc = numel(find(DS.TL == classified))/numel(DS.TL);
fprintf('TRANSFERRED accuracy for class %d: %f\n', c2, new_acc);



