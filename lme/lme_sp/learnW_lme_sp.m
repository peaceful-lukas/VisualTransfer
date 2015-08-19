function W = learnW_lme_sp(DS, W, U, param)

    n = 0;

    while( n < param.maxIterW )
        tic
        triplets = sampleTriplets(DS, W, U, param);
        dW = computeGradient(DS, W, U, triplets, param);
        W = update(W, dW, param.lr_W);
        % W = update(W, dW, param.lr_W/(1 + n * param.lr_W));
        
        if mod(n, 100) == 99;
            fprintf('W) iter %d / ', n+1);
            loss = getSampleLoss(DS, W, U, triplets, param);
            fprintf('elapsed time: %f\n', toc);
        end
        n = n + 1;
    end
end

function W = update(W, dW, learning_rate)
    W = W - learning_rate * dW;
end

function dW = computeGradient(DS, W, U, triplets, param)
    X = DS.D;
    lowDim = param.lowDim;
    featureDim = param.featureDim;
    lambda_W = param.lambda_W;
    numTriplets = size(triplets, 1);

    if( numTriplets > 0 )
        dW_cell = arrayfun(@(n) (U(:, triplets(n, 3)) - U(:, triplets(n, 2)))*X(:, triplets(n, 1))', 1:numTriplets, 'UniformOutput', false);
        dW_cat = cat(3, dW_cell{:});
        dW = sum(dW_cat, 3);
        
        dW = dW/param.batchSize + lambda_W*W/size(W, 2);
    else
        dW = lambda_W*W/size(W, 2);
    end
end


function loss = getSampleLoss(DS, W, U, triplets, param)
    X = DS.D;
    numClasses = param.numClasses;
    lambda_W = param.lambda_W;
    lambda_U = param.lambda_U;
    triplets = sampleTriplets(DS, W, U, param);
    numTriplets = size(triplets, 1);

    err = 0;
    if numTriplets > 0
        err = sum(arrayfun(@(n) param.lm + X(:, triplets(n, 1))'*W'*(U(:, triplets(n, 3)) - U(:, triplets(n, 2))), 1:numTriplets));
        err = err/numTriplets;
    end

    loss = err + lambda_W*0.5*norm(W, 'fro')^2 + lambda_U*0.5*norm(U, 'fro')^2;
    fprintf('viol: %d / loss: %f / cErr: %f / nomrW: %f / normU: %f / ', numTriplets, loss, err, norm(W, 'fro'), norm(U, 'fro'));
end

function triplets = sampleTriplets(DS, W, U, param)
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

    triplets = [dataIdx corrLabels incorrLabels];
    triplets = validTriplets(DS, W, U, triplets, param);
end

function triplets = validTriplets(DS, W, U, triplets, param)
    X = DS.D;
    numTriplets = size(triplets, 1);

    val = arrayfun(@(n) param.lm + X(:, triplets(n, 1))'*W'*(U(:, triplets(n, 3)) - U(:, triplets(n, 2))), 1:numTriplets);
    valids = find(val > 0);
    triplets = triplets(valids, :);
end
