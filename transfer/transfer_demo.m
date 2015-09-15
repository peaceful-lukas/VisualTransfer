
[tPairs S] = transferPairs(U, param);

param0 = param;
param_new = param;

U0 = U;
U_new = U;

for i=1:size(tPairs, 1)
    fprintf('\n\n\n================================ class %d ================================\n', i);

    c1 = tPairs(i, 1);
    c2 = tPairs(i, 2);
    scale_alpha = 1.0;

    [U_new, param_new, ~, matched_pairs, trainTargetClasses] = transfer(DS, W, U_new, U0, c1, c2, scale_alpha, param_new, param0);


    % Locally train
    param_new.lambda_U_local = 10;
    param_new.lr_U_local = 0.001;
    [U_new param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses);
    % [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses);

    transfer_dispAccuracies(DS, W, U, U_new, param_new.numPrototypes, param0);
end





