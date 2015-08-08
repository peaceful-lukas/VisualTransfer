function affinityMatrix = computeAffinityMatrix(U1, U2, matches)
% vertex similarity - inner product
% affinity score    - multiplication

    vertexSimMatrix = zeros(size(U1, 2), size(U2, 2));
    scores = sum(U1(:, matches(:, 1)).*U2(:, matches(:, 2)), 1);
    idx = sub2ind(size(vertexSimMatrix), matches(:, 1), matches(:, 2));
    vertexSimMatrix(idx) = scores;

    numMatches = size(matches, 1);
    affinityMatrix = scores'*scores;
end