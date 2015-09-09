function [W U] = local_train(DS, W, U, trainTargetClasses, param)

U = learnU_new(DS, W, U, param);
[~, accuracy] = dispAccuracy(method, n+1, DS, W, U, param);

