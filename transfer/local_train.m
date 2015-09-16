function [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses)

% regenerate classification triplets
fprintf('Local Training LME ... \n');
param_new.cTriplets = generateClassificationTriplets(DS, param_new);
param_new.sTriplets = generateLocasStructurePreservingTriplets(param_new, trainTargetClasses);

% locally learn prototypes (U)
W_retrained = local_learnW(DS, W, U_new, param_new, trainTargetClasses);
U_retrained = local_learnU(DS, W, U_new, param_new, trainTargetClasses);

fprintf('Local LME RESULT\n');
fprintf('\tbefore\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_new, param_new);
fprintf('\tafter\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_retrained, param_new);
fprintf('\n\n\n\n');