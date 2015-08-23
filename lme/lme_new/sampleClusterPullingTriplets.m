function pTriplets = sampleClusterPullingTriplets(DS, W, U, param)
% (i, k, l), the indices regard the actual permutation in U

num_pTriplets = size(param.pTriplets, 1);
pTriplets = param.pTriplets;

loss = param.p_lm + diag((W*X(:, pTriplets(:, 1)))' * (U(:, pTriplets(:, 3)) - U(:, pTriplets(:, 2))));
valids = find(loss > 0);
pTriplets = pTriplets(valids, :);
