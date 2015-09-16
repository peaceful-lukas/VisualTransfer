function W = local_learnW(DS, W, U, param, trainTargetClasses)

W_orig = W;

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    cTriplets = sampleClassificationTriplets(DS, W, U, param);

    dW = computeGradient(DS, W, W_orig, U, cTriplets, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('W%d) ', n);
        loss = local_sampleLoss(DS, W, U, W_orig, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end


% update
function W = update(W, dW, param)

W = W - param.lr_W_local * dW;



% gradient computation
function dW = computeGradient(DS, W, W_orig, U, cTriplets, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    c_dW = (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))) * X(:, cTriplets(:, 1))';
    c_dW = c_dW/param.c_batchSize;
end

bal_c = param.bal_c/(param.bal_c + param.lambda_W_local);
lambda_W_local = param.lambda_W_local/(param.bal_c + param.lambda_W_local);

dW = bal_c*c_dW + lambda_W_local*(W - W_orig);
