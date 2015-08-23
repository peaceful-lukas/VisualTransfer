function sTriplets = sampleStructurePreservingTriplets(DS, W, U, param)

% (k, k', l), the indices regard the actual permutation in U

num_sTriplets = size(param.sTriplets, 1);
sTriplets = param.sTriplets(randperm(num_sTriplets, param.s_batchSize), :);

loss = param.s_lm + diag( U(:, sTriplets(:, 1))' * (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2))) );
valids = find(loss > 0);
sTriplets = sTriplets(valids, :);

