function protoRepr(DS, W, U, classNum, param, showImg)


protoStartIdx = [0 cumsum(param.numPrototypes)];

class_feat_idx = find(DS.DL == classNum); % class feature idx
[classified_score, classified_idx] = max(DS.D(:, class_feat_idx)'*W'*U, [], 2);

for n=protoStartIdx(classNum)+1:protoStartIdx(classNum+1)
    classified_proto_idx = find(classified_idx == n);
    
    if length(classified_proto_idx) > 0
        classified_proto_score = classified_score(classified_proto_idx);
        [~, classified_proto_score_sorted_idx] = sort(classified_proto_score, 'descend');
        classified_proto_idx_sorted = classified_proto_idx(classified_proto_score_sorted_idx);

        repr_feat_idx = class_feat_idx(classified_proto_idx_sorted);

        fprintf('class %d) prototype %d (count %d)', classNum, n-protoStartIdx(classNum), length(repr_feat_idx));

        if showImg
            figure;
            for cnt=1:min(3, length(repr_feat_idx))
                imshow(DS.DI{repr_feat_idx(cnt)});
                pause;
            end
            fprintf('\n');
            % pause;
        else
            fprintf(' - ');
            for cnt=1:min(3, length(repr_feat_idx))
                fprintf('%6d', repr_feat_idx(cnt));
            end
            fprintf('\n');
        end
    else
        fprintf('class %d) prototype %d  - NOTHING\n', classNum, n-protoStartIdx(classNum));
    end
end
