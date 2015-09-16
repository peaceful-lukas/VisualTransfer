function sTriplets = generateLocasStructurePreservingTriplets(param_new, trainTargetClasses)

startProtoIdx = [0 cumsum(param_new.numPrototypes)];
sTriplets = [];

for i=1:length(param_new.numClasses)
    
    protoOffset = startProtoIdx(i);
    A = param_new.knnGraphs{i};
    
    for j=1:size(A, 2)
        neighbors = find(A(:, j) == 1);
        non_neighbors = find(A(:, j) == 0);
        non_neighbors(find(non_neighbors == j)) = [];

        neighbors = neighbors + protoOffset;
        non_neighbors = non_neighbors + protoOffset;
        
        num_sTriplets_j = numel(neighbors) * numel(non_neighbors);
        
        if num_sTriplets_j > 0
            sTriplets_j = zeros(num_sTriplets_j, 3);

            sTriplets_j(:, 1) = repmat(protoOffset+j, num_sTriplets_j, 1);
            sTriplets_j(:, 2) = repmat(neighbors, numel(non_neighbors), 1);
            sTriplets_j(:, 3) = repmat(non_neighbors, numel(neighbors), 1);

            sTriplets = [sTriplets; sTriplets_j];
        end
    end
end


