function [spTriplets knnGraphs] = generate_sp_triplets_by_crp(P, param)

    numClasses = param.numClasses;
    numPrototypes = param.numPrototypes;
    knn_const = param.knn_const;

    spTriplets = [];
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
                sim = sum(bsxfun(@times, P_c, P_c(:, k)), 1);
                sim(k) = -Inf;

                [~, sorted_sim_idx] = sort(sim, 'descend');
                
                % knn-graph
                knnGraphs{c}(k, sorted_sim_idx(1:knn_const)) = 1;

                % sp-triplets
                sorted_sim_idx = sorted_sim_idx + proto_offset;
                neighbors_idx = sorted_sim_idx(1:knn_const);

                numSpTriplets_ck = (knn_const) * (numPrototypes(c)-1-knn_const);

                spTriplets_ck = zeros(numSpTriplets_ck, 3);
                tmp_sec_col = repmat(neighbors_idx, numPrototypes(c)-1-knn_const, 1);
                spTriplets_ck(:, 1) = repmat(proto_offset+k, numSpTriplets_ck, 1);
                spTriplets_ck(:, 2) = tmp_sec_col(:);
                spTriplets_ck(:, 3) = repmat(sorted_sim_idx(knn_const+1:end-1)', knn_const, 1);

                spTriplets = [spTriplets; spTriplets_ck];
            end
        end

        knnGraphs{c} = max(triu(knnGraphs{c})+triu(knnGraphs{c})', tril(knnGraphs{c}) + tril(knnGraphs{c})');
        proto_offset = proto_offset + numPrototypes(c);
    end

    spTriplets;
    knnGraphs;
end