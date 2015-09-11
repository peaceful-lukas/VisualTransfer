function acc_list = transfer_dispAccuracies(DS, W, U, U_new, new_numPrototypes, param)
acc_list = [];
for cls = 1:param.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W, U, param);
    new_acc = getNewAccuracy(cls, DS, W, U_new, new_numPrototypes, param);
    acc_list = [acc_list; orig_acc new_acc];
    fprintf('Accuracy (class %d) : %.4f ----> %.4f\n', cls, orig_acc, new_acc);    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W, U0, param)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U0, [], 2);
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
