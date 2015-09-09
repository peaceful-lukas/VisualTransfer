
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

class_trainsfer_pairs = [maxS_idx' (1:12)']; % ------> (transfer direction)


scale_alpha = 1.1;
for i=1:size(class_trainsfer_pairs, 1)
    c1 = class_trainsfer_pairs(i, 1);
    c = class_trainsfer_pairs(i, 2);
    scale_alpha = 1.1;

    [U_new U new_numPrototypes trainTargetClasses] = transfer(DS, W, U, c1, c2, scale_alpha, param);

    param.numPrototypes = new_numPrototypes;
    U = U_new;
end

