function voting = probabilisticVoting(match, U1, U2, param)
% Find voting score using probability distribution for matching prototypes with direction U1 -> U2

    knn1 = param.knn1;
    knn2 = param.knn2;
    voting_alpha = param.voting_alpha;
    matchAnchor1 = U1(:, match(1));
    matchAnchor2 = U2(:, match(2));


    % find the k-Nearest Neighbors from the match anchor.
    knnIdx1 = knnSearch(matchAnchor1, U1, knn1);
    knnIdx2 = knnSearch(matchAnchor2, U2, knn2);

    % compute voting scores.
    scores_raw = exp(voting_alpha*U1(:, knnIdx1)'*U2(:, knnIdx2));
    scores = bsxfun(@rdivide, scores_raw, sum(scores_raw, 2));

    voting = zeros(size(U1, 2), size(U2, 2));
    voting(knnIdx1, knnIdx2) = scores;
end


function knnIdx = knnSearch(anchor, pointSet, k)
    scores = anchor'*pointSet;
    [scores_sorted, sorted_idx] = sort(scores, 'descend');
    knnIdx = sorted_idx(1:min(k, length(sorted_idx)));
end