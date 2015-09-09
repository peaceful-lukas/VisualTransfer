
% L = {};
% for i=1:param.numClasses
%     L{i} = laplacian(param.knnGraphs{i}, 1); % normalized
% end

% M = zeros(param.numClasses, param.numClasses); % prematched score matrix by graph laplacians
% for i=1:param.numClasses
%     for j=1:param.numClasses
%         M(i, j) = prematching(L{i}, L{j});
%     end
% end



protoStartIdx = [0 cumsum(param.numPrototypes)];
S = zeros(param.numClasses, param.numClasses); % Similarity between class prototype distributions

for i=1:param.numClasses
    for j=1:param.numClasses
        if i == j
            S(i, j) = -Inf;
        else
            S(i, j) = mean(U(:, protoStartIdx(i)+1:protoStartIdx(i+1)), 2)'*mean(U(:, protoStartIdx(j)+1:protoStartIdx(j+1)), 2);
        end
    end
end

[maxS, maxS_idx] = max(S, [], 1)

transferPairs = [maxS_idx' (1:12)']; % ------> (transfer direction)


scale_alpha = 1.1;
for i=1:size(transferPairs, 1)
    c1 = transferPairs(i, 1);
    c2 = transferPairs(i, 2);

    [U_new, param_new, ~, matched_pairs, trainTargetClasses] = transfer(DS, W, U, c1, c2, scale_alpha, param);


    % Locally train
    U_retrained = local_train(DS, W, U_new, param_new, trainTargetClasses);

    % update
    param = param_new;
    U = U_new;
end





