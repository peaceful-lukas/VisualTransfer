function cTriplets = generateClassificationTriplets(DS, param)

cTriplets = [];

numPrototypes = param.numPrototypes;
numClasses = param.numClasses;
numData = length(DS.DL);
protoStartIdx = [0 cumsum(numPrototypes)]; % protoStartIdx(n)+1:protoStartIdx(n+1)
totalProto = protoStartIdx(end);

for c=1:numClasses
    protoIdx_c = protoStartIdx(c)+1:protoStartIdx(c+1);
    dataIdx_c = find(DS.DL == c);
    numData_c = length(dataIdx_c);
    
    numTripletsPerInstance = numPrototypes(c)*(totalProto-numPrototypes(c));
    cTriplets_c = zeros(numData_c*numTripletsPerInstance, 3);

    repCorr = reshape(repmat(protoIdx_c, totalProto-numPrototypes(c), 1), numTripletsPerInstance, 1);
    incorr = 1:totalProto;
    incorr(protoIdx_c) = [];
    repIncorr = repmat(incorr', numPrototypes(c), 1);
    for i=1:numData_c
        cTriplets_c((i-1)*numTripletsPerInstance+1:i*numTripletsPerInstance, 1) = dataIdx_c(i);
        cTriplets_c((i-1)*numTripletsPerInstance+1:i*numTripletsPerInstance, 2) = repCorr;
        cTriplets_c((i-1)*numTripletsPerInstance+1:i*numTripletsPerInstance, 3) = repIncorr;
    end

    cTriplets = [cTriplets; cTriplets_c];
end

cTriplets;
