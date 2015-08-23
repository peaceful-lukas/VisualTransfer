function pTriplets = generateClusterPullingTriplets(protoAssign, numPrototypes)

% (i, k, l), the indices regard the actual permutation in U
pTriplets = [];
numData = length(protoAssign);
protoStartIdx = [0 cumsum(numPrototypes)] % protoStartIdx(n)+1:protoStartIdx(n+1)

for c=1:length(numPrototypes)
    protoIdx = protoStartIdx(c)+1:protoStartIdx(c+1);
    dataIdx = find(protoAssign >= protoIdx(1) & protoAssign <= protoIdx(end));
    numProtoIdx = length(protoIdx);
    numDataIdx = length(dataIdx);
    pTriplets_c = zeros(numDataIdx*(numProtoIdx-1), 3);

    numTripletsPerData = numProtoIdx-1;
    for i=1:numDataIdx
        incorr = 1:numProtoIdx;
        incorr(find(protoAssign(dataIdx))) = [];
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, :) = zeros(numTripletsPerData, 3);
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 1) = dataIdx(i);
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 2) = protoAssign(dataIdx(i));
        pTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 3) = incorr;
    end

    pTriplets = [pTriplets; pTriplets_c];
end


pTriplets;