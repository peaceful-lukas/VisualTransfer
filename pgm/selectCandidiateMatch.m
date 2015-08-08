function cand_matches = selectCandidiateMatch(voting_space, U1, U2, param)

    [idx_p1, idx_p2, voting_scores] = find(voting_space);
    [tmp, iv] = sort(voting_scores, 'descend');

    tmp = find(isinf(tmp), 1, 'last');
    nSelMatch = min(tmp, length(iv));
    sel_iv = iv(1:nSelMatch);

    for n=nSelMatch+1:length(iv)
        cur_iv = iv(n);
        sim = U1(:, idx_p1(cur_iv))'*U2(:, idx_p2(cur_iv));
        if sim > param.match_sim_thrsh
            sel_iv = [sel_iv; cur_iv];
        end
    end

    cand_matches = [idx_p1(sel_iv) idx_p2(sel_iv)];
end