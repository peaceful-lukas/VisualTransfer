function U = learnU_new(DS, W, U, param)

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    pTriplets = sampleClusterPullingTriplets(DS, W, U, param);
    sTriplets = sampleStructurePreservingTriplets(DS, W, U, param);

    dU = computeGradient(DS, WX, U, cTriplets, pTriplets, sTriplets, aux, param);
    U = update(U, dU, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('U%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end

% update
function U = update(U, dU, param)

U = U - param.lr_U * dU;


% gradient computation
function dU = computeGradient(DS, WX, U, cTriplets, pTriplets, sTriplets, aux, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);
num_pTriplets = size(pTriplets, 1);
num_sTriplets = size(sTriplets, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    c_dU = WX(:, cTriplets(:, 1))*(aux(:, cTriplets(:, 3)) - aux(:, cTriplets(:, 2)))';
    c_dU = c_dU/param.c_batchSize;
end

p_dU = zeros(size(U));
if num_pTriplets > 0
    p_dU = WX(:, pTriplets(:, 1))*(aux(:, pTriplets(:, 3)) - aux(:, pTriplets(:, 2)))';
    p_dU = p_dU/param.p_batchSize;
end

s_dU = zeros(size(U));
if num_sTriplets > 0
    s1 = (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 1))';
    s2 = U(:, sTriplets(:, 1))*aux(:, sTriplets(:, 2))';
    s3 = U(:, sTriplets(:, 1))*aux(:, sTriplets(:, 3))';
    s_dU = s1 + s2 + s3;
    s_dU = s_dU/param.s_batchSize;
end

dU = c_dU + p_dU + s_dU + param.lambda_U*U;

