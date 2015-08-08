function [matches sim_scores] = initial_match(U1, U2, param)
% Similarity Measure
%   - dot product
% 
% Input
%   U1, U2 - column-major matrix of prototypes.
% 
% Output 
%   matches - column-major matrix of pairs of prototypes.

    match_thrsh = param.match_thrsh;

    len1 = size(U1, 2);
    len2 = size(U2, 2);
    matches = zeros(2, len1*len2);
    sim_scores = zeros(1, len1*len2);
    % simfunc = param.simfunc;

    match_count = 0;
    for i=1:len1
        all_scores = sum(bsxfun(@times, U2, U1(:, i)), 1);
        matched_idx = find(all_scores > match_thrsh);
        current_matches = [repmat(i, 1, numel(matched_idx)); matched_idx];
        current_sim_scores = all_scores(matched_idx);

        matches(:, match_count+1:match_count+numel(matched_idx)) = current_matches;
        sim_scores(match_count+1:match_count+numel(matched_idx)) = current_sim_scores;
        match_count = match_count + numel(matched_idx);
    end

    matches = matches(:, 1:match_count)';
    sim_scores = sim_scores(1:match_count);
end

