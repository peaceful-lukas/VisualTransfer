function U = learnU_lme_sp(DS, W, U, param)

    n = 0;

    while( n < param.maxIterU )
        tic
        triplets = sampleTriplets(DS, W, U, param);
        dU = computeGradient(DS, W, U, triplets, param);
        U = update(U, dU, param.lr_U);
        % U = update(U, dU, param.lr_U/(1 + n * param.lr_U));
        
        if mod(n, 100) == 99;
            fprintf('U) iter %d / ', n+1);
            loss = getSampleLoss(DS, W, U, triplets, param);
            fprintf('elapsed time: %f\n', toc);
        end
        n = n + 1;
    end
end

function U = update(U, dU, learning_rate)
    U = U - learning_rate * dU;
end

function dU = computeGradient(DS, W, U, triplets, param)
    X = DS.D;
    lowDim = param.lowDim;
    lambda_U = param.lambda_U;
    numClasses = param.numClasses;
    numTriplets = size(triplets, 1);

    if( numTriplets > 0 )
        dU = zeros(lowDim, numClasses);
        for n=1:numTriplets
            x_i = X(:, triplets(n, 1));
            y_i = triplets(n, 2);
            c = triplets(n, 3);

            bin_y_i = zeros(numClasses, 1);
            bin_c = zeros(numClasses, 1);
            bin_y_i(y_i) = 1;
            bin_c(c) = 1;

            dU = dU + W*x_i*(bin_c - bin_y_i)';
        end

        dU = dU/param.batchSize + lambda_U*U/size(U, 2);
    else
        dU = lambda_U*U/size(U, 2);
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
