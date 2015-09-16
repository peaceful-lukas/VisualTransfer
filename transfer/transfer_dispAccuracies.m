function acc_list = transfer_dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0);
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
