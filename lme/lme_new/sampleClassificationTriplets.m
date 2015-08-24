function cTriplets = sampleClassificationTriplets(DS, W, U, param)

% (i, y_i, c)

X = DS.D;
num_cTriplets = size(param.cTriplets, 1);
cTriplets = param.cTriplets(randperm(num_cTriplets, param.c_batchSize), :);

loss = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
valids = find(loss > 0);
cTriplets = cTriplets(valids, :);
