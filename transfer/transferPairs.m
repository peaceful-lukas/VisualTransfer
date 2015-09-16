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

% S_tmp = S;
% S_tmp(find(S_tmp < 0.7)) = 0



% % take it exponetially
% sim_thrsh = 0.9;
% S_exp = exp(S)./repmat(max(exp(S), [], 1), size(S, 1), 1);
% S_exp(find(S_exp < sim_thrsh)) = 0;
% % S_exp = exp(S./repmat(max(S, [], 1), size(S, 1), 1));
% % S = exp(S);

% % Best match
% % [maxS, maxS_idx] = max(S, [], 1);
% % tPairs = [maxS_idx' (1:param.numClasses)']; % ------> (transfer direction)

% % All match
% sim_thrsh = 0.
% find(S < 




% graph Laplacian between classes
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
