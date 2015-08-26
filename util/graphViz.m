function graphViz(U, classNum, param, orig, classProtos)

protoStartIdx = [0 cumsum(param.numPrototypes)];

if ~orig
    U_c = U(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    [coeff,reduced_U_c, eigval] = princomp(U_c');
    reduced_U_c = reduced_U_c(:, 1:2);
    gplot(param.knnGraphs{classNum}, reduced_U_c)
else
    org_U_c = classProtos(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    [coeff,reduced_U_c, eigval] = princomp(org_U_c');
    reduced_U_c = reduced_U_c(:, 1:2);
    gplot(param.knnGraphs{classNum}, reduced_U_c)
end


