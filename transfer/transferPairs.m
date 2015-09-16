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

S_sorted = sort(S(:), 'descend');

S_tmp = S;
% S_tmp(find(S_tmp < 0.5)) = 0;
S_tmp(find(S_tmp < S_sorted(param.numClasses))) = 0;

% transfer direction : <-------- ( but not important since S is a symmetric matrix.)
tPairs = [floor((find(S_tmp)-1)/param.numClasses)+1 mod(find(S_tmp), param.numClasses)];
tPairs(find(tPairs == 0)) = param.numClasses;

% Print out the transfer directions determined.
fprintf('TRANSFER DIRECTIONS\n');
fprintf('<------------------\n');
str_tPairs = stringifyClasses(tPairs, param.dataset)


% % take it exponetially
% sim_thrsh = 0.9;
% S_exp = exp(S)./repmat(max(exp(S), [], 1), size(S, 1), 1);
% S_exp(find(S_exp < sim_thrsh)) = 0;
% % S_exp = exp(S./repmat(max(S, [], 1), size(S, 1), 1));
% % S = exp(S);

% % Best match
% [maxS, maxS_idx] = max(S, [], 1);
% tPairs = [maxS_idx' (1:param.numClasses)']; % ------> (transfer direction)


function str_tPairs = stringifyClasses(tPairs, dataset)

str_tPairs = cell(size(tPairs));

if strcmp(dataset, 'awa')
    [idx, clsname] = textread('/v9/AwA/raw/Animals_with_Attributes/classes.txt', '%d %s');
    
    for i=1:size(tPairs, 1)
        str_tPairs{i, 1} = clsname{tPairs(i, 1)};
        str_tPairs{i, 2} = clsname{tPairs(i, 2)};
    end
elseif strcmp(dataset, 'pascal3d_pascal') || strcmp(dataset, 'pascal3d_imagenet') || strcmp(dataset, 'pascal3d_all')
    clsnames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
    
    for i=1:size(tPairs, 1)
        for j=1:2
            str_tPairs{i, j} = clsnames{tPairs(i, j)};
        end
    end
else
    fprintf('\nno class name list on %s\n\n', dataset);
end







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

