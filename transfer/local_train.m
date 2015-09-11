function [U_retrained param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses)

% regenerate classification triplets
fprintf('... generating classification triplets for local training .. ')
keyboard;
param_new.cTriplets = generateClassificationTriplets(DS, param_new);

% locally learn prototypes (U)
U_retrained = local_learnU(DS, W, U_new, param_new, trainTargetClasses);


fprintf('before\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_new, param_new);
fprintf('after\n');
[~, accuracy] = dispAccuracy('lme_new', 0, DS, W, U_retrained, param_new);







function U_retrained = local_learnU(DS, W, U, param, trainTargetClasses)

U_orig = U;

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU/10;
    cTriplets = sampleClassificationTriplets(DS, W, U, param);

    dU = computeGradient(WX, U, U_orig, aux, cTriplets, param, trainTargetClasses);
    U = update(U, dU, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('U%d) ', n);
        loss = sampleLossForLocal(DS, W, U, U_orig, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end

U_retrained = U;




% update
function U = update(U, dU, param)

U = U - param.lr_U * dU;


% gradient computation
function dU = computeGradient(WX, U, U_orig, aux, cTriplets, param, trainTargetClasses)

num_cTriplets = size(cTriplets, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    c_dU = WX(:, cTriplets(:, 1))*(aux(:, cTriplets(:, 3)) - aux(:, cTriplets(:, 2)))';
    c_dU = c_dU/param.c_batchSize;
end

dU = param.bal_c*c_dU + param.lambda_U_local*(U - U_orig);



% loss function
function loss = sampleLossForLocal(DS, W, U, U_orig, param)


X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);

cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    cErr_vec = diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
    viol = find(cErr_vec > 0);
    num_cV = length(viol);

    if viol > 0
        cErr = param.c_lm + sum(cErr_vec(viol));
    end
end

loss = cErr + param.lambda_U_local*0.5*norm(U - U_orig, 'fro')^2;
fprintf('cV: %d / cErr: %f / normU: %f / ', num_cV, cErr, norm(U, 'fro'));







% function sTriplets = localStructurePreservingTriplets(param_new, trainTargetClasses)

% startProtoIdx = [0 cumsum(param_new.numPrototypes)];
% sTriplets = [];

% for i=1:length(param_new.numClasses)
    
%     protoOffset = startProtoIdx(i);
%     A = param_new.knnGraphs{i};
    
%     for j=1:size(A, 2)
%         neighbors = find(A(:, j) == 1);
%         non_neighbors = find(A(:, j) == 0);
%         non_neighbors(find(non_neighbors == j)) = [];

%         neighbors = neighbors + protoOffset;
%         non_neighbors = non_neighbors + protoOffset;
        
%         num_sTriplets_j = numel(neighbors) * numel(non_neighbors);
        
%         if num_sTriplets_j > 0
%             sTriplets_j = zeros(num_sTriplets_j, 3);

%             sTriplets_j(:, 1) = repmat(protoOffset+j, num_sTriplets_j, 1);
%             sTriplets_j(:, 2) = repmat(neighbors, numel(non_neighbors), 1);
%             sTriplets_j(:, 3) = repmat(non_neighbors, numel(neighbors), 1);

%             sTriplets = [sTriplets; sTriplets_j];
%         end
%     end
% end


