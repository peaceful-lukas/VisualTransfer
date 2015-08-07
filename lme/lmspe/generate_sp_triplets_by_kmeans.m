function spTriplets = generate_sp_triplets_by_kmeans(Cent, param, showGraph)
%   prototype_offset : numbers indicating how many prototypes before regarding the current class being dealt with.

    numClasses = param.numClasses;
    numPrototypes = param.numPrototypes;
    knn_const = param.knn_const;

    spTriplets = [];

    if showGraph == 1
        A = {};
    end

    % generate knn graph
    for c=1:numClasses
        
        if showGraph == 1
            A{c} = zeros(numPrototypes, numPrototypes);
        end

        Cent_c = Cent(:, (c-1)*numPrototypes+1:c*numPrototypes);
        prototype_offset = (c-1)*numPrototypes;

        for k=1:numPrototypes
            sim = sum(bsxfun(@times, Cent_c, Cent_c(:, k)), 1);
            sim(k) = -Inf;

            [~, sorted_sim_idx] = sort(sim, 'descend');
            sorted_sim_idx = sorted_sim_idx + prototype_offset;

            neighbors_idx = sorted_sim_idx(1:knn_const);

            if showGraph == 1
                A{c}(neighbors_idx(1:end), k) = 1;
            end

            numSpTriplets_ck = (knn_const) * (numPrototypes-1-knn_const);
            
            spTriplets_ck = zeros(numSpTriplets_ck, 3);
            tmp_sec_col = repmat(neighbors_idx, numPrototypes-1-knn_const, 1);
            spTriplets_ck(:, 1) = repmat(prototype_offset+k, numSpTriplets_ck, 1);
            spTriplets_ck(:, 2) = tmp_sec_col(:);
            spTriplets_ck(:, 3) = repmat(sorted_sim_idx(knn_const+1:end-1)', knn_const, 1);

            spTriplets = [spTriplets; spTriplets_ck];
            % fprintf('%d triplets added.\n', size(spTriplets_ck, 1));
        end

        if showGraph == 1
            imagesc(A);
            pause;
        end
    end


    spTriplets;
end