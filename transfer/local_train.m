function U_retrained = local_train(DS, W, U_new, param_new, trainTargetClasses)

sTriplets_local = localStructurePreservingTriplets(param_new, trainTargetClasses);
param_new.sTriplets = sTriplets_local;

U_retrained = local_learnU(DS, W, U_new, param_new);
fprintf('before\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_new, param_new);
fprintf('after\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_retrained, param_new);



function sTriplets_local = localStructurePreservingTriplets(param_new, trainTargetClasses)

startProtoIdx = [0 cumsum(param_new.numPrototypes)];
sTriplets_local = [];

for i=1:length(trainTargetClasses)
    cls = trainTargetClasses(i);
    protoOffset = startProtoIdx(cls);

    A = param_new.knnGraphs{cls};
    
    for j=1:length(size(A, 2))
        neighbors = find(A(:, j) == 1);
        non_neighbors = find(A(:, j) == 0);
        non_neighbors(find(non_neighbors == j)) = [];

        neighbors = neighbors + protoOffset;
        non_neighbors = non_neighbors + protoOffset;
        
        num_sTriplets_local_j = numel(neighbors) * numel(non_neighbors);
        
        if num_sTriplets_local_j > 0
            sTriplets_local_j = zeros(num_sTriplets_local_j, 3);

            sTriplets_local_j(:, 1) = repmat(protoOffset+j, num_sTriplets_local_j, 1);
            sTriplets_local_j(:, 2) = repmat(neighbors, numel(non_neighbors), 1);
            sTriplets_local_j(:, 3) = repmat(non_neighbors, numel(neighbors), 1);

            sTriplets_local = [sTriplets_local; sTriplets_local_j];
        end
    end
end


