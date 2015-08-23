function [X_sol candidate_matches score_GM] = progressiveGraphMatching(U1, U2, param)
% revised Progressive Graph Matching
% ( Graph matching after sp-LME )
% 
% Inputs
%   U1 (U2) - graph of class prototypes
% 
% Outputs
%   X_sol : n x n 
% 
% written by Taewoo Kim. (Aug 2, 2015)
% 

    maxIterGM = param.maxIterGM;

    % find initial matches
    [candidate_matches_new, scores] = initial_match(U1, U2, param);
    affinityMatrix_new = computeAffinityMatrix(U1, U2, candidate_matches_new);
    [group1, group2] = make_group12(candidate_matches_new);

    % progressive graph matching algorithm start
    stopFlag = 0;
    score_GM = 0;
    for iterGM = 1:maxIterGM
        
        %----- Graph Matching ------
        X_raw_new = RRWM(affinityMatrix_new, group1, group2);
        X_sol_new = greedyMapping(X_raw_new, group1, group2);
        score_GM_new = X_sol_new'*affinityMatrix_new*X_sol_new;

        if score_GM_new <= score_GM
            iterGM = iterGM - 1;
            stopFlag = 1;
        else
            X_sol = X_sol_new;
            X_raw = X_raw_new;
            score_GM = score_GM_new;
            candidate_matches = candidate_matches_new;
            affinityMatrix = affinityMatrix_new;
        end

        %----- Evaluate the solutions -----
                %%%% from now on, '_new' postfix indicates ones that will be used in the next iteration.
        match_idx = find(X_sol);
        match_list = candidate_matches(match_idx, :);
        match_score = X_raw(match_idx);

        %---- Check to Continue ----
        if iterGM == maxIterGM, stopFlag = 1; end;
        if stopFlag
            fprintf('>>>>>>>>>>>>>> reached the max score.\n');
            break;
        end
        

        %---- Graph Progression ----
        voting_space = zeros(size(U1, 2), size(U2, 2));

        for iter_m = 1:length(match_idx)

            % forward voting
            match = match_list(iter_m, :);
            voting = probabilisticVoting(match, U1, U2, param);
            voting_space = voting_space + voting;

            % backward voting
            match = fliplr(match);
            voting = probabilisticVoting(match, U2, U1, param);
            voting_space = voting_space + voting';
        end

        % make sure that the current matches are included in the next iteration.
        for iter_m = 1:length(match_idx)
            match = match_list(iter_m, :);
            voting_space(match(1), match(2)) = Inf;
        end

        % select new candidate matches
        candidate_matches_new = selectCandidiateMatch(voting_space, U1, U2, param);

        % compute new affinity matrix
        affinityMatrix_new = computeAffinityMatrix(U1, U2, candidate_matches_new);
        [group1, group2] = make_group12(candidate_matches_new);

        fprintf('iter %d\n', iterGM);
    end


end


