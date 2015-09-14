function [tPairs S] = transferPairs(U, param)


protoStartIdx = [0 cumsum(param.numPrototypes)];
S = zeros(param.numClasses, param.numClasses); % Similarity between class prototype distributions

for i=1:param.numClasses
    for j=1:param.numClasses
        if i == j
            S(i, j) = -Inf;
        else
            % maximum score
            S(i, j) = max(max(U(:, protoStartIdx(i)+1:protoStartIdx(i+1))'*U(:, protoStartIdx(j)+1:protoStartIdx(j+1))));

            % average score
            % S(i, j) = mean(U(:, protoStartIdx(i)+1:protoStartIdx(i+1)), 2)'*mean(U(:, protoStartIdx(j)+1:protoStartIdx(j+1)), 2);
        end
    end
end

[maxS, maxS_idx] = max(S, [], 1);

tPairs = [maxS_idx' (1:12)']; % ------> (transfer direction)



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
