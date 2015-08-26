function nearestEmbeddings(DS, W, U, classNum, param, showImg)

protoStartIdx = [0 cumsum(param.numPrototypes)];
U_c = U(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));

for n=1:param.numPrototypes(classNum)
    u_proto = U_c(:, n);
    [~, idx] = sort(DS.D'*W'*u_proto, 'descend');
    fprintf('prototype %d   ', n);
    if showImg
        figure;
        imshow(DS.DI{idx(1)});
        fprintf('\n');
        pause;
    else
        fprintf(': %d\n', idx(1));
    end
end

