function cTriplets = sampleClassificationTriplets(DS, W, U, param)

% (i, y_i, c)
numData = length(DS.DL);
numClasses = param.numClasses;
c_batchSize = param.c_batchSize;

% randomly choose data indices
dataIdx = randperm(numData, c_batchSize)';

% the correct labels of the data sampled.
corrLabels = reshape(DS.DL(dataIdx), length(dataIdx), 1);

% randomly choose incorrect labels of the data sampled.
incorrLabels = ceil(numClasses*rand(c_batchSize, 1));
while length(find(incorrLabels == corrLabels)) > 0
    collapsed = find(incorrLabels == corrLabels);
    numCollapsed = length(collapsed);
    incorrLabels(collapsed) = ceil(numClasses*rand(numCollapsed, 1));
end

% initial cTriplets
cTriplets = [dataIdx corrLabels incorrLabels];


% check whether valid triplets or not.
X = DS.D;
numPrototypes = param.numPrototypes;
protoStartIdx = [0 cumsum(numPrototypes)];
num_cTriplets = size(cTriplets, 1);

loss = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
valids = find(loss > 0);

% return cTriplets
cTriplets = cTriplets(valids, :);