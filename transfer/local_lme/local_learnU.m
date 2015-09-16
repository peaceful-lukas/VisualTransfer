function U = local_learnU(DS, W, U, param, trainTargetClasses)

U_orig = U;

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU;
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    sTriplets = sampleStructurePreservingTriplets(DS, W, U, param);

    dU = computeGradient(WX, U, U_orig, aux, cTriplets, param, trainTargetClasses);
    U = update(U, dU, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('U%d) ', n);
        loss = local_sampleLoss(DS, W, U, W, U_orig, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end

U_retrained = U;




% update
function U = update(U, dU, param)

U = U - param.lr_U_local * dU;


% gradient computation
function dU = computeGradient(WX, U, U_orig, aux, cTriplets, sTriplets, param, trainTargetClasses)

num_cTriplets = size(cTriplets, 1);
num_sTriplets = size(sTriplets, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    c_dU = WX(:, cTriplets(:, 1))*(aux(:, cTriplets(:, 3)) - aux(:, cTriplets(:, 2)))';
    c_dU = c_dU/param.c_batchSize;
end

s_dU = zeros(size(U));
if num_sTriplets > 0
    s1 = (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 1))';
    s2 = U(:, sTriplets(:, 1))*aux(:, sTriplets(:, 2))';
    s3 = U(:, sTriplets(:, 1))*aux(:, sTriplets(:, 3))';
    s_dU = s1 + s2 + s3;
    s_dU = s_dU/param.s_batchSize;
end

dU = param.bal_c*c_dU + param.bal_s*s_dU + param.lambda_U_local*U;


