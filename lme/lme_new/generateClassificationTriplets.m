function cTriplets = generateClassificationTriplets(DS, param)

numData = length(DS.DL);
numClasses = param.numClasses;

cTriplets = [];
for c=1:numClasses
    dataIdx_c = find(DS.DL == c);
    numData_c = length(dataIdx_c);
    
    numTripletsPerData = numClasses-1;
    cTriplets_c = zeros(numData_c*numTripletsPerData, 3);

    for i=1:numData_c
        incorr = 1:numClasses;
        incorr(c) = [];
        cTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 1) = dataIdx_c(i);
        cTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 2) = c;
        cTriplets_c((i-1)*numTripletsPerData+1:i*numTripletsPerData, 3) = incorr;
    end

    cTriplets = [cTriplets; cTriplets_c];
end

cTriplets;
