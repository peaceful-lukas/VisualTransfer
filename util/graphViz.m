function graphViz(U, classNum, param, orig, classProtos)

lab_num = 1:param.numPrototypes(classNum);
lab_txt = cell(1, max(lab_num));
for n=lab_num
    lab_txt(n) = {num2str(lab_num(n))};
end


protoStartIdx = [0 cumsum(param.numPrototypes)];

if ~orig
    U_c = U(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    [coeff,reduced_U_c, eigval] = princomp(U_c');
    reduced_U_c = reduced_U_c(:, 1:2);
    gplotwl(param.knnGraphs{classNum}, reduced_U_c,lab_txt);
else
    org_U_c = classProtos(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    [coeff,reduced_U_c, eigval] = princomp(org_U_c');
    reduced_U_c = reduced_U_c(:, 1:2);
    gplotwl(param.knnGraphs{classNum}, reduced_U_c,lab_txt);
end


