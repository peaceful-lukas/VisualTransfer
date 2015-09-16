function [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses)

% regenerate classification triplets
fprintf('... generating classification triplets for local training .. \n');
param_new.cTriplets = generateClassificationTriplets(DS, param_new);
param_new.sTriplets = generateLocasStructurePreservingTriplets(param_new, trainTargetClasses);

% locally learn prototypes (U)
W_retrained = local_learnW(DS, W, U_new, param_new, trainTargetClasses);
U_retrained = local_learnU(DS, W, U_new, param_new, trainTargetClasses);


fprintf('before\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_new, param_new);
fprintf('after\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_retrained, param_new);

