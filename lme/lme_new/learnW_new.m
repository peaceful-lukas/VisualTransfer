function W = learnW_new(DS, W, U, param)

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    pTriplets = sampleClusterPullingTriplets(DS, W, U, param);

    dW = computeGradient(DS, W, U, cTriplets, pTriplets, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('W%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end


% update
function W = update(W, dW, param)

W = W - param.lr_W * dW;



% gradient computation
function dW = computeGradient(DS, W, U, cTriplets, pTriplets, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);
num_pTriplets = size(pTriplets, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    c_dW = (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))) * X(:, cTriplets(:, 1))';
    c_dW = c_dW/param.c_batchSize;
end

p_dW = zeros(size(W));
if num_pTriplets > 0
    p_dW = (U(:, pTriplets(:, 3)) - U(:, pTriplets(:, 2))) * X(:, pTriplets(:, 1))';
    p_dW = p_dW/param.p_batchSize;
end

dW = c_dW + p_dW + param.lambda_W*W;