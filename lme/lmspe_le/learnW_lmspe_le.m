function W = learnW_lmspe_le(DS, W, U, M, param)

    n = 0;

    while( n < param.maxIterW )
        tic
        cTriplets = sampleClassficationTriplets(DS, W, U, M, param);
        dW = computeGradient(DS, W, U, M, cTriplets, param);
        [W param] = update(W, dW, param.lr_W, param);

        if mod(n, 100) == 99
            fprintf('W) iter %d / ', n+1);
            loss = getSampleLoss(DS, W, U, M, param);
            fprintf('elapsed time: %f\n', toc);
        end
        n = n + 1;
    end
end

function [W param] = update(W, dW, learning_rate, param)
    if norm(dW) == Inf || isnan(norm(dW))
        param.alpha = param.alpha/2;
        fprintf('softmax parameter(alpha) scale decrease: alpha = %f\n', param.alpha);
        W;
    else
        W = W - learning_rate * dW;
    end
end

function dW = computeGradient(DS, W, U, M, cTriplets, param)
    X = DS.D;
    lambda_W = param.lambda_W;
    num_cTriplets = size(cTriplets, 1);
    
    if( num_cTriplets > 0 )
        dW = computeApproximateMaxGradient(X, W, U, M, cTriplets, 3, param) - computeApproximateMaxGradient(X, W, U, M, cTriplets, 2, param);
        dW = dW/num_cTriplets + lambda_W*W/size(W, 2);
    else
        dW = lambda_W*W/size(W, 2);
    end
end

function appr_max_dW = computeApproximateMaxGradient(X, W, U, M, triplets, tripl_col, param)
    numPrototypes = param.numPrototypes;
    numTriplets = size(triplets, 1);
    featureDim = param.featureDim;
    alpha = param.alpha;

    D = arrayfun(@(n) sum(exp(alpha*X(:, triplets(n, 1))'*W'*U*M(:, numPrototypes*(triplets(n, tripl_col)-1)+1:numPrototypes*triplets(n, tripl_col)))), 1:numTriplets);

    N = arrayfun(@(n) sum(repmat(exp(alpha*X(:, triplets(n, 1))'*W'*U*M(:, numPrototypes*(triplets(n, tripl_col)-1)+1:numPrototypes*triplets(n, tripl_col))), featureDim, 1).*M(:, numPrototypes*(triplets(n, tripl_col)-1)+1:numPrototypes*triplets(n, tripl_col)), 2), 1:numTriplets, 'UniformOutput', false);
    N = cat(2, N{:});

    appr_max_dW_cell = arrayfun(@(n) U*N(:, n)*X(:, triplets(n, 1))'/D(n), 1:numTriplets, 'UniformOutput', false);
    appr_max_dW_cat = cat(3, appr_max_dW_cell{:});
    appr_max_dW = sum(appr_max_dW_cat, 3);
end

function loss = getSampleLoss(DS, W, U, M, param)
    X = DS.D;
    numPrototypes = param.numPrototypes;
    cTriplets = sampleClassficationTriplets(DS, W, U, M, param);
    spTriplets = validStructurePreservingTriplets(U, M, param);
    num_cTriplets = size(cTriplets, 1);
    num_spTriplets = size(spTriplets, 1);
    lambda_W = param.lambda_W;
    lambda_U = param.lambda_U;


    cErr = 0;
    if( num_cTriplets > 0 )
        incorrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U*M(:, numPrototypes*(cTriplets(n, 3)-1)+1:numPrototypes*cTriplets(n, 3))), 1:num_cTriplets);
        corrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U*M(:, numPrototypes*(cTriplets(n, 2)-1)+1:numPrototypes*cTriplets(n, 2))), 1:num_cTriplets);
        cErr = sum(param.c_lm + incorrs - corrs)/num_cTriplets;
        cErr = param.bal_c*cErr;
    end

    spErr = 0;
    if( num_spTriplets > 0 )
        spErr = sum(param.sp_lm + sum((U*M(:, spTriplets(:, 1))).*(U*(M(:, spTriplets(:, 3)) - M(:, spTriplets(:, 2)))), 1))/num_spTriplets;
        spErr = param.bal_sp*spErr;
    end

    loss = cErr + spErr + lambda_W*0.5*norm(W, 'fro')^2/size(W, 2) + lambda_U*0.5*norm(U, 'fro')^2/size(U, 2);

    fprintf('cViol: %d / spViol: %d / loss: %f / cErr: %f / spErr: %f / normW: %f / normU: %f / ', num_cTriplets, num_spTriplets, loss, cErr, spErr, norm(W, 'fro'), norm(U, 'fro'));
end

function valid_spTriplets = validStructurePreservingTriplets(U, M, param)
    spTriplets = param.spTriplets;
    vals = param.sp_lm + sum((U*M(:, spTriplets(:, 1))).*(U*M(:, spTriplets(:, 3))), 1) - sum((U*M(:, spTriplets(:, 1))).*(U*M(:, spTriplets(:, 2))), 1);
    valids = find(vals > 0);
    valid_spTriplets = spTriplets(valids, :);
end

function cTriplets = sampleClassficationTriplets(DS, W, U, M, param)
    numData = numel(DS.DL);
    numClasses = param.numClasses;
    batchSize = param.batchSize;

    % randomly sample data indices
    dataIdx = ceil(numData * rand(batchSize, 1));
    
    % the correct labels of the sampled data
    corrLabels = DS.DL(dataIdx);

    % randomly choose incorrect labels of the sampled data
    incorrLabels = ceil(numClasses * rand(batchSize, 1));
    collapsed = find(incorrLabels == corrLabels);
    incorrLabels(collapsed) = mod(incorrLabels(collapsed)+1, numClasses+1);
    incorrLabels(find(incorrLabels == 0)) = 1;

    cTriplets = [dataIdx corrLabels incorrLabels];
    cTriplets = validClassificationTriplets(DS, W, U, M, cTriplets, param);
end

function cTriplets = validClassificationTriplets(DS, W, U, M, cTriplets, param)
    X = DS.D;
    numPrototypes = param.numPrototypes;
    num_cTriplets = size(cTriplets, 1);
   
    incorrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U*M(:, numPrototypes*(cTriplets(n, 3)-1)+1:numPrototypes*cTriplets(n, 3))), 1:num_cTriplets);
    corrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U*M(:, numPrototypes*(cTriplets(n, 2)-1)+1:numPrototypes*cTriplets(n, 2))), 1:num_cTriplets);
    valids = find(param.c_lm + incorrs - corrs > 0);
    % fprintf('num valids(%d) : ', numel(valids));
   
    cTriplets = cTriplets(valids, :);    
end