function cTriplets = sampleClassificationTriplets(DS, W, U, param)

% (i, y_i, c)
X = DS.D;
i_vec = ceil(numel(DS.DL) * rand(param.c_batchSize, 1));
yi_vec = DS.DL(i_vec);
c_vec = generateDifferentClassList(yi_vec, param.numClasses)

cTriplets = [i_vec yi_vec c_vec];

loss = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
valids = find(loss > 0);
cTriplets = cTriplets(valids, :);


function c_vec = generateDifferentClassList(yi_vec, numClasses)

c_vec = ceil(numClasses * rand(length(yi_vec), 1));
collapsed = find(yi_vec == c_vec);
c_vec = mod(c_vec(collapsed), numClasses);
c_vec(find(c_vec == 0)) = numClasses;
