function U = learnU_lmspe_crp(DS, W, U, param)

    n = 0;

    while( n < param.maxIterU )
        tic
        cTriplets = sampleClassficationTriplets(DS, W, U, param);
        spTriplets = validStructurePreservingTriplets(U, param);
        dU = computeGradient(DS, W, U, cTriplets, spTriplets, param);
        [U param] = update(U, dU, param.lr_U, param);

        if mod(n, 100) == 99
            fprintf('U) iter %d / ', n+1);
            loss = getSampleLoss(DS, W, U, param);
            fprintf('elapsed time: %f\n', toc);
        end
        n = n + 1;
    end
end

function [U param] = update(U, dU, learning_rate, param)
    if norm(dU) == Inf || isnan(norm(dU))
        param.alpha = param.alpha/2;
        fprintf('softmax parameter(alpha) scale decrease: alpha = %f\n', param.alpha);
        U;
    else
        U = U - learning_rate * dU;
    end
end

function dU = computeGradient(DS, W, U, cTriplets, spTriplets, param)
    X = DS.D;
    num_cTriplets = size(cTriplets, 1);
    num_spTriplets = size(spTriplets, 1);
    numClasses = param.numClasses;
    numPrototypes = param.numPrototypes;
    protoStartIdx = [0 cumsum(numPrototypes)];
    alpha = param.alpha;
    lambda_U = param.lambda_U;
    lowDim = param.lowDim;
    bal_c = param.bal_c;
    bal_sp = param.bal_sp;


    c_dU = zeros(size(U, 1), size(U, 2));
    if( num_cTriplets > 0 )
        for n=1:num_cTriplets
            expUc = exp(alpha*X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 3))+1:protoStartIdx(cTriplets(n, 3)+1)));
            Dc = sum(expUc);
            dUc = repmat(expUc, lowDim, 1).*repmat((W*X(:, cTriplets(n, 1))), 1, numPrototypes(cTriplets(n, 3)))/Dc;

            expUyi = exp(alpha*X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 2))+1:protoStartIdx(cTriplets(n, 2)+1)));
            Dyi = sum(expUyi);
            dUyi = repmat(expUyi, lowDim, 1).*repmat((W*X(:, cTriplets(n, 1))), 1, numPrototypes(cTriplets(n, 2)))/Dyi; 

            trip_dU = zeros(size(U, 1), size(U, 2));
            trip_dU(:, protoStartIdx(cTriplets(n, 3))+1:protoStartIdx(cTriplets(n, 3)+1)) = dUc;
            trip_dU(:, protoStartIdx(cTriplets(n, 2))+1:protoStartIdx(cTriplets(n, 2)+1)) = -dUyi;
            
            c_dU = c_dU + trip_dU;
        end
        
        c_dU = c_dU/param.batchSize;
    end


    sp_dU = zeros(size(U, 1), size(U, 2));
    if( num_spTriplets > 0 )
        for n=1:num_spTriplets % complicating to vectorize due to the collapsed prototype indices.
            sp_dU(:, spTriplets(n, 1)) = sp_dU(:, spTriplets(n, 1)) + U(:, spTriplets(n, 3)) - U(:, spTriplets(n, 2));
            sp_dU(:, spTriplets(n, 2)) = sp_dU(:, spTriplets(n, 2)) - U(:, spTriplets(n, 1));
            sp_dU(:, spTriplets(n, 3)) = sp_dU(:, spTriplets(n, 3)) + U(:, spTriplets(n, 1));
        end
        sp_dU = sp_dU/size(param.spTriplets, 1);
    end

    if norm(sp_dU, 'fro') > 0
        ratio = sqrt(norm(c_dU, 'fro')/norm(sp_dU, 'fro'));
        sp_dU = ratio*sp_dU;
    end
    
    dU = bal_c*c_dU + bal_sp*sp_dU + lambda_U*U/size(U, 2);
end


function loss = getSampleLoss(DS, W, U, param)
    X = DS.D;
    numPrototypes = param.numPrototypes;
    protoStartIdx = [0 cumsum(numPrototypes)];
    cTriplets = sampleClassficationTriplets(DS, W, U, param);
    spTriplets = validStructurePreservingTriplets(U, param);
    num_cTriplets = size(cTriplets, 1);
    num_spTriplets = size(spTriplets, 1);
    lambda_W = param.lambda_W;
    lambda_U = param.lambda_U;
    

    cErr = 0;
    if( num_cTriplets > 0 )
        incorrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 3))+1:protoStartIdx(cTriplets(n, 3)+1))), 1:num_cTriplets);
        corrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 2))+1:protoStartIdx(cTriplets(n, 2)+1))), 1:num_cTriplets);
        cErr = sum(param.c_lm + incorrs - corrs)/num_cTriplets;
        cErr = param.bal_c*cErr;
    end

    spErr = 0;
    if( num_spTriplets > 0 )
        spErr = sum(param.sp_lm + sum(U(:, spTriplets(:, 1)).*(U(:, spTriplets(:, 3)) - U(:, spTriplets(:, 2))), 1))/num_spTriplets;
        spErr = param.bal_sp*spErr;
    end

    loss = cErr + spErr + lambda_W*0.5*norm(W, 'fro')^2/size(W, 2) + lambda_U*0.5*norm(U, 'fro')^2/size(U, 2);

    fprintf('cViol: %d / spViol: %d / loss: %f / cErr: %f / spErr: %f / normW: %f / normU: %f / ', num_cTriplets, num_spTriplets, loss, cErr, spErr, norm(W, 'fro'), norm(U, 'fro'));
end

function valid_spTriplets = validStructurePreservingTriplets(U, param)
    spTriplets = param.spTriplets;
    vals = param.sp_lm + sum(U(:, spTriplets(:, 1)).*U(:, spTriplets(:, 3)), 1) - sum(U(:, spTriplets(:, 1)).*U(:, spTriplets(:, 2)), 1);
    valids = find(vals > 0);
    valid_spTriplets = spTriplets(valids, :);
end

function cTriplets = sampleClassficationTriplets(DS, W, U, param)
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
    cTriplets = validClassificationTriplets(DS, W, U, cTriplets, param);
end


function cTriplets = validClassificationTriplets(DS, W, U, cTriplets, param)
    X = DS.D;
    numPrototypes = param.numPrototypes;
    protoStartIdx = [0 cumsum(numPrototypes)];
    num_cTriplets = size(cTriplets, 1);
   
    incorrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 3))+1:protoStartIdx(cTriplets(n, 3)+1))), 1:num_cTriplets);
    corrs = arrayfun(@(n) max(X(:, cTriplets(n, 1))'*W'*U(:, protoStartIdx(cTriplets(n, 2))+1:protoStartIdx(cTriplets(n, 2)+1))), 1:num_cTriplets);
    valids = find(param.c_lm + incorrs - corrs > 0);
    % fprintf('num valids(%d) : ', numel(valids));
   
    cTriplets = cTriplets(valids, :);    
end
