function loss = sampleLoss(DS, W, U, param)

X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);
pTriplets = sampleClusterPullingTriplets(DS, W, U, param);
sTriplets = sampleStructurePreservingTriplets(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);
num_pTriplets = size(pTriplets, 1);
num_sTriplets = size(sTriplets, 1);

cErr = 0;
if num_cTriplets > 0
    cErr = param.c_lm + sum(diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2)))));
end

pErr = 0;
if num_pTriplets > 0
   pErr = param.p_lm + sum(diag((W*X(:, pTriplets(:, 1)))' * (U(:, pTriplets(:, 3)) - U(:, pTriplets(:, 2)))));
end

sErr = 0;
if num_sTriplets > 0
    sErr = param.s_lm + sum(diag( U(:, sTriplets(:, 1))' * (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2))) ));
end

loss = cErr + pErr + sErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('cV: %d / pV: %d / sV: %d / cE: %f / pE: %f / sE: %f / normW: %f / normU: %f / ', num_cTriplets, num_pTriplets, num_sTriplets, cErr, pErr, sErr, norm(W, 'fro'), norm(U, 'fro'));

