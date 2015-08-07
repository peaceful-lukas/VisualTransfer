function [train_acc test_acc] = dispAccuracy(method, iter, DS, W, U, param, numPrototypes)
    if strcmp(method, 'lme_sp')
        [~, classified] = max(DS.D'*W'*U, [], 2);
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('Alternation %d) TRAIN data set accuracy : %.4f\n', iter, train_acc);

        [~, classified] = max(DS.T'*W'*U, [], 2);
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('Alternation %d) TEST data set accuracy : %.4f\n', iter, test_acc);

    elseif strcmp(method, 'lmspe')
        [~, classified] = max(DS.D'*W'*U, [], 2);
        classified = ceil(classified/param.numPrototypes);
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('Alternation %d) train set accuracy : %.4f\n', iter, train_acc);

        [~, classified] = max(DS.T'*W'*U, [], 2);
        classified = ceil(classified/param.numPrototypes);
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('Alternation %d) TEST set accuracy :  %.4f\n', iter, test_acc);

    elseif strcmp(method, 'lmspe_crp')

        cumNumProto = cumsum(param.numPrototypes);
        [~, classified_raw] = max(DS.D'*W'*U, [], 2);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('Alternation %d) train set accuracy : %.4f\n', iter, train_acc);


        cumNumProto = cumsum(param.numPrototypes);
        [~, classified_raw] = max(DS.T'*W'*U, [], 2);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('Alternation %d) TEST set accuracy :  %.4f\n', iter, test_acc);


    elseif strcmp(method, 'lmspe_le')
        [~, classified] = max(DS.D'*W'*U*M, [], 2);
        classified = ceil(classified/param.numPrototypes);
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('Alternation %d) train set accuracy : %.4f\n', iter, train_acc);

        [~, classified] = max(DS.T'*W'*U*M, [], 2);
        classified = ceil(classified/param.numPrototypes);
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('Alternation %d) TEST set accuracy :  %.4f\n', iter, test_acc);
    end
end