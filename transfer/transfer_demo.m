% GRAPH MATCHING SCORES

% MatchingScores = zeros(param.numClasses, param.numClasses);
% for i=1:param.numClasses
%     for j=i+1:param.numClasses
%         c1 = i;
%         c2 = j;


%         numProto_c1 = param.numPrototypes(c1);
%         numProto_c2 = param.numPrototypes(c2);
%         protoStartIdx = [0 cumsum(param.numPrototypes)];
%         U_c1 = U(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
%         U_c2 = U(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

%         simMatrix = U_c1'*U_c2;
%         sim_scores = sort(simMatrix(:), 'descend');

%         param_gm.maxIterGM = 10;
%         param_gm.match_thrsh = sim_scores(min(numProto_c1, numProto_c2));
%         param_gm.match_sim_thrsh = sim_scores(max(numProto_c1, numProto_c2));
%         param_gm.knn1 = 3;
%         param_gm.knn2 = 4;
%         param_gm.voting_alpha = 10;


%         [X_sol candidate_matches score_GM] = progGM(U_c1, U_c2, param_gm);
%         MatchingScores(i, j) = score_GM;
%     end
% end






%%%%%%%%%%%%%%%%


[tPairs S] = transferPairs(U, param);
% tPairs([2 5 6 10], :) = []
param0 = param;
param_new = param;

W0 = W;
W_new = W;

U0 = U;
U_new = U;

for i=1:size(tPairs, 1)
    fprintf('\n\n\n================================ class %d ================================\n', i);

    % Transfer
    c1 = tPairs(i, 1);
    c2 = tPairs(i, 2);
    scale_alpha = 0.7;
    [U_new, param_new, matched_pairs, trainTargetClasses, score_GM] = transfer(DS, W_new, U_new, W0, U0, c1, c2, scale_alpha, param_new, param0);


    % % Locally train
    param_new.lambda_W_local = 0.1;
    param_new.lambda_U_local = 10;
    param_new.lr_U_local = 0.00001;
    param_new.lr_W_local = 0.00001;
    param_new.bal_c = 1;
    param_new.bal_s = 1;
    [W_new, U_new param_new] = local_train(DS, W_new, U_new, param_new, trainTargetClasses);

    % Result
    transfer_dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0);
end








