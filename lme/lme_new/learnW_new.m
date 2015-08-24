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
    % c_dW = (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))) * X(:, cTriplets(:, 1))';
    % c_dW = c_dW/param.c_batchSize;

    c_dW_cell = arrayfun(@(n) (U(:, cTriplets(n, 3)) - U(:, cTriplets(n, 2)))*X(:, cTriplets(n, 1))', 1:num_cTriplets, 'UniformOutput', false);
    c_dW_cat = cat(3, c_dW_cell{:});
    c_dW = sum(c_dW_cat, 3);
    
    c_dW = c_dW/param.c_batchSize;
end

p_dW = zeros(size(W));
if num_pTriplets > 0
    % p_dW = (U(:, pTriplets(:, 3)) - U(:, pTriplets(:, 2))) * X(:, pTriplets(:, 1))';
    % p_dW = p_dW/param.p_batchSize;

    p_dW_cell = arrayfun(@(n) (U(:, pTriplets(n, 3)) - U(:, pTriplets(n, 2)))*X(:, pTriplets(n, 1))', 1:num_pTriplets, 'UniformOutput', false);
    p_dW_cat = cat(3, p_dW_cell{:});
    p_dW = sum(p_dW_cat, 3);

    p_dW = p_dW/param.p_batchSize;
end

dW = param.bal_c*c_dW + param.bal_p*p_dW + param.lambda_W*W;

