function pTriplets = sampleClusterPullingTriplets(DS, W, U, param)

% (i, k, l), the indices regard the actual permutation in U

X = DS.D;
protoStartIdx = [0 cumsum(numPrototypes)]; % protoStartIdx(n)+1:protoStartIdx(n+1)

i_vec = ceil(numel(DS.DL) * rand(param.c_batchSize, 1));
k_vec = param.protoAssign(i_vec);
% l_vec = generateDifferentPrototypeList()

% c_vec = generateDifferentClassList(yi_vec, param.numClasses)

% cTriplets = [i_vec yi_vec c_vec];

% num_pTriplets = size(param.pTriplets, 1);
% pTriplets = param.pTriplets(randperm(num_pTriplets, param.p_batchSize), :);

loss = param.p_lm + diag((W*X(:, pTriplets(:, 1)))' * (U(:, pTriplets(:, 3)) - U(:, pTriplets(:, 2))));
valids = find(loss > 0);
pTriplets = pTriplets(valids, :);

% pTriplets = [];






pTriplets = [];
numData = length(protoAssign);
protoStartIdx = [0 cumsum(numPrototypes)]; % protoStartIdx(n)+1:protoStartIdx(n+1)

for c=1:length(numPrototypes)
    protoIdx = protoStartIdx(c)+1:protoStartIdx(c+1);
    dataIdx = find(protoAssign >= protoIdx(1) & protoAssign <= protoIdx(end));
    numProtoIdx = length(protoIdx);
    numDataIdx = length(dataIdx);
    
    numTripletsPerData = numProtoIdx-1;
    pTriplets_c = zeros(numDataIdx*numTripletsPerData, 3);
    for i=1:numDataIdx
        incorr = 1:numProtoIdx;
        incorr(find(protoAssign(dataIdx(i)))) = [];
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 1) = dataIdx(i);
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 2) = protoAssign(dataIdx(i));
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 3) = incorr;
    end
    
    pTriplets = [pTriplets; pTriplets_c];
end


pTriplets;