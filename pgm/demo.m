
addpath 'RRWM'


param.maxIterGM = 10; % progressive graph matching maximum iteration
param.match_thrsh = 0.005; % initial match threshold
param.match_sim_thrsh = 0.0005; % match threshold
param.knn1 = 3;
param.knn2 = 6;
param.voting_alpha = 10; % scales the voting score (the higher the value, the more distinct the scores.)


[X_sol, cand_matches] = progGM(U1, U2, param);

matched_pairs = cand_matches(find(X_sol), :);



% %-------------------- Subgraph Transfer --------------------
% numMatched = size(matched_pairs, 1);
% uniq_p1 = unique(matched_pairs(:, 1));
% uniq_p2 = unique(matched_pairs(:, 2));

% %--- Transfer from U1 to U2 ---
% not_matched_p1 = 1:10;
% not_matched_p1(uniq_p1) = [];

% randIdx = randi(numel(not_matched_p1));
% targetTransferIdx = not_matched_p1(randIdx);

% transferred_prototype = zeros(100, 1);
% for n=1:numMatched
%     transferred_prototype = transferred_prototype + U2(:, matched_pairs(n, 2)) - U1(:, matched_pairs(n, 1)) + U1(:, targetTransferIdx);
% end
% transferred_prototype = transferred_prototype/numMatched;

