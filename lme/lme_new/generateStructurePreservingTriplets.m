function [sTriplets knnGraphs] = generateStructurePreservingTriplets(P, param)


numClasses = param.numClasses;
numPrototypes = param.numPrototypes;
knn_const = param.knn_const;

sTriplets = [];
knnGraphs = {};

% centering
proto_offset = 0;
for c=1:numClasses
    P_c = P(:, proto_offset+1:proto_offset+numPrototypes(c));
    P(:, proto_offset+1:proto_offset+numPrototypes(c)) = bsxfun(@minus, P_c,  mean(P_c, 2));
    proto_offset = proto_offset + numPrototypes(c);
end


% generate knn graph
proto_offset = 0;
for c=1:numClasses
    knnGraphs{c} = zeros(numPrototypes(c), numPrototypes(c));
    P_c = P(:, proto_offset+1:proto_offset+numPrototypes(c));

    % in case that the number of prototypes for a class 'c' is less than knn_const, which is generally 3
    if knn_const >= numPrototypes(c)
        knnGraphs{c} = ones(numPrototypes(c), numPrototypes(c)) - eye(numPrototypes(c), numPrototypes(c));
        
    else
        for k=1:numPrototypes(c)
            dst = sum(bsxfun(@minus, P_c, P_c(:, k)).^2, 1);
            dst(k) = Inf;
            [~, sorted_idx] = sort(dst, 'ascend');

            % sim = sum(bsxfun(@times, P_c, P_c(:, k)), 1);
            % sim(k) = -Inf;
            % [~, sorted_idx] = sort(sim, 'descend');
            
            % knn-graph
            knnGraphs{c}(k, sorted_idx(1:knn_const)) = 1;

            % sp-triplets
            sorted_idx = sorted_idx + proto_offset;
            neighbors_idx = sorted_idx(1:knn_const);

            numSpTriplets_ck = (knn_const) * (numPrototypes(c)-1-knn_const);

            sTriplets_ck = zeros(numSpTriplets_ck, 3);
            tmp_sec_col = repmat(neighbors_idx, numPrototypes(c)-1-knn_const, 1);
            sTriplets_ck(:, 1) = repmat(proto_offset+k, numSpTriplets_ck, 1);
            sTriplets_ck(:, 2) = tmp_sec_col(:);
            sTriplets_ck(:, 3) = repmat(sorted_idx(knn_const+1:end-1)', knn_const, 1);

            sTriplets = [sTriplets; sTriplets_ck];
        end
    end

    knnGraphs{c} = max(triu(knnGraphs{c})+triu(knnGraphs{c})', tril(knnGraphs{c}) + tril(knnGraphs{c})');
    proto_offset = proto_offset + numPrototypes(c);
end

sTriplets;
knnGraphs;



