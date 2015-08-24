function param = getParam(method, dataset)
    param = {};

    switch dataset
        case 'pascal'
            param = getPascalParam(method);
        
        case 'pascal3d'
            param = getPascal3dParam(method);
        
        case 'awa'
            param = getAwaParam(method);

        case '3dobj'
            param = get3DObjParam(method);

        case 'coil100'
            param = getCoil100Param(method);
        
        otherwise
            error(['No such dataset(' dataset ')']);
    end

    param.dataset = dataset;
end



function param = getPascalParam(method)

    if strcmp(method, 'lme_sp')
        % BEST 80.xx %
        param.numClasses = 20;
        param.lowDim = 100;
        param.featureDim = 4096;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 20;
        param.batchSize = 100; % mini-batch size

        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.lm = 0.1; % large margin for classification
        param.lambda_W = 10; % regularizer coefficient
        param.lambda_U = 10; % regularizer coefficient
    elseif strcmp(method, 'lmspe')

    elseif strcmp(method, 'lmspe_crp')

    elseif strcmp(method, 'lmspe_le')

    elseif strcmp(method, 'lme_new')
        
    end

end


function param = getPascal3dParam(method)
    
    if strcmp(method, 'lme_sp')
        param.numClasses = 12;
        param.lowDim = 12;
        param.featureDim = 9216;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 20;
        param.batchSize = 50; % mini-batch size

        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.lm = 10; % large margin for classification
        param.lambda_W = 100; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient

    elseif strcmp(method, 'lmspe')

    elseif strcmp(method, 'lmspe_crp')
        % SAMPLE
        param.numClasses = 12;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 10; % mini-batch size
        param.lowDim = 200;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.01; % large margin for structure preserving
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 1000; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 10;

    elseif strcmp(method, 'lmspe_le')

    elseif strcmp(method, 'lme_new')
        param.numClasses = 12;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.c_batchSize = 1000;
        param.p_batchSize = 100;
        param.s_batchSize = 100;
        param.lowDim = 80;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 100; % large margin for classification
        param.p_lm = 10; % large margin for classification
        param.s_lm = 1; % large margin for structure preserving
        param.lambda_W = 0.1; % regularizer coefficient
        param.lambda_U = 1; % regularizer coefficient
        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.0001; % learning rate for U
        param.bal_c = 1;
        param.bal_p = 1;
        param.bal_s = 1;
    end

end


function param = getAwaParam(method)
    
    if strcmp(method, 'lme_sp')
        % BEST 69.xx%
        param.numClasses = 50;
        param.lowDim = 50;
        param.featureDim = 9216;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 30; % mini-batch size

        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.lm = 20; % large margin for classification
        param.lambda_W = 100; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient

    elseif strcmp(method, 'lmspe')
        % BEST 67.93%
        param.numClasses = 50;
        param.numPrototypes = 10;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 10; % mini-batch size
        param.lowDim = 100;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.01; % large margin for structure preserving
        param.lambda_W = 1000; % regularizer coefficient
        param.lambda_U = 1000; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00005; % learning rate for W
        param.lr_U = 0.00005; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 30;

    elseif strcmp(method, 'lmspe_crp')

        % (1)
        % param.numClasses = 50;
        % param.maxIterW = 1000;
        % param.maxIterU = 1000;
        % param.maxAlter = 50;
        % param.batchSize = 10; % mini-batch size
        % param.lowDim = 100;
        % param.featureDim = 9216;

        % param.knn_const = 3; % constant for constructing k-nn graph.
        % param.c_lm = 10; % large margin for classification
        % param.sp_lm = 0.1; % large margin for structure preserving
        % param.lambda_W = 100000; % regularizer coefficient
        % param.lambda_U = 100; % regularizer coefficient
        % param.alpha = 5; % softmax parameter.
        % param.lr_W = 0.00001; % learning rate for W
        % param.lr_U = 0.00001; % learning rate for U
        % param.bal_c = 1;
        % param.bal_sp = 10;

        % (2)
        % param.numClasses = 50;
        % param.maxIterW = 1000;
        % param.maxIterU = 1000;
        % param.maxAlter = 50;
        % param.batchSize = 10; % mini-batch size
        % param.lowDim = 300;
        % param.featureDim = 9216;

        % param.knn_const = 3; % constant for constructing k-nn graph.
        % param.c_lm = 10; % large margin for classification
        % param.sp_lm = 0.1; % large margin for structure preserving
        % param.lambda_W = 1000000; % regularizer coefficient
        % param.lambda_U = 10000; % regularizer coefficient
        % param.alpha = 5; % softmax parameter.
        % param.lr_W = 0.00001; % learning rate for W
        % param.lr_U = 0.00001; % learning rate for U
        % param.bal_c = 1;
        % param.bal_sp = 10;

        % (3)
        % param.numClasses = 50;
        % param.maxIterW = 1000;
        % param.maxIterU = 1000;
        % param.maxAlter = 50;
        % param.batchSize = 10; % mini-batch size
        % param.lowDim = 400;
        % param.featureDim = 9216;

        % param.knn_const = 3; % constant for constructing k-nn graph.
        % param.c_lm = 10; % large margin for classification
        % param.sp_lm = 0.1; % large margin for structure preserving
        % param.lambda_W = 1000000; % regularizer coefficient
        % param.lambda_U = 10000; % regularizer coefficient
        % param.alpha = 5; % softmax parameter.
        % param.lr_W = 0.00001; % learning rate for W
        % param.lr_U = 0.00001; % learning rate for U
        % param.bal_c = 1;
        % param.bal_sp = 10;

        % (4)
        % param.numClasses = 50;
        % param.maxIterW = 1000;
        % param.maxIterU = 1000;
        % param.maxAlter = 50;
        % param.batchSize = 10; % mini-batch size
        % param.lowDim = 100;
        % param.featureDim = 9216;

        % param.knn_const = 3; % constant for constructing k-nn graph.
        % param.c_lm = 10; % large margin for classification
        % param.sp_lm = 0.1; % large margin for structure preserving
        % param.lambda_W = 100000; % regularizer coefficient
        % param.lambda_U = 100; % regularizer coefficient
        % param.alpha = 5; % softmax parameter.
        % param.lr_W = 0.000001; % learning rate for W
        % param.lr_U = 0.000001; % learning rate for U
        % param.bal_c = 1;
        % param.bal_sp = 10;

        % (5) % same with the (1)
        % param.numClasses = 50;
        % param.maxIterW = 1000;
        % param.maxIterU = 1000;
        % param.maxAlter = 50;
        % param.batchSize = 10; % mini-batch size
        % param.lowDim = 100;
        % param.featureDim = 9216;

        % param.knn_const = 3; % constant for constructing k-nn graph.
        % param.c_lm = 10; % large margin for classification
        % param.sp_lm = 0.1; % large margin for structure preserving
        % param.lambda_W = 100000; % regularizer coefficient
        % param.lambda_U = 100; % regularizer coefficient
        % param.alpha = 5; % softmax parameter.
        % param.lr_W = 0.00001; % learning rate for W
        % param.lr_U = 0.00001; % learning rate for U
        % param.bal_c = 1;
        % param.bal_sp = 10;



        (6) % lr_W and lr_U  --> 1/10
        param.numClasses = 50;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 20; % mini-batch size
        param.lowDim = 100;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.1; % large margin for structure preserving
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.000001; % learning rate for W
        param.lr_U = 0.000001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 10;


    elseif strcmp(method, 'lmspe_le')
        param.numClasses = 50;
        param.numPrototypes = 10;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 30;
        param.batchSize = 50; % mini-batch size
        param.lowDim = 100;
        param.featureDim = 9216;


        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 0.1; % large margin for classification
        param.sp_lm = 0.001; % large margin for structure preserving
        param.lambda_W = 10; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 1;
    

    elseif strcmp(method, 'lme_new')
        
    end

end

function param = get3DObjParam(method)
    if strcmp(method, 'lme_sp')
        param.numClasses = 10;
        param.lowDim = 10;
        param.featureDim = 9216;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 30; % mini-batch size

        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.lm = 20; % large margin for classification
        param.lambda_W = 100; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient
    elseif strcmp(method, 'lmspe')

    elseif strcmp(method, 'lmspe_crp')
        param.numClasses = 10;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 10; % mini-batch size
        param.lowDim = 50;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.01; % large margin for structure preserving
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 1000; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 10;
    elseif strcmp(method, 'lmspe_le')

    end
end

function param = getCoil100Param(method)
    if strcmp(method, 'lme_sp')
        param.numClasses = 100;
        param.lowDim = 100;
        param.featureDim = 9216;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 30; % mini-batch size

        param.lr_W = 0.0001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.lm = 20; % large margin for classification
        param.lambda_W = 100; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient
    elseif strcmp(method, 'lmspe')

    elseif strcmp(method, 'lmspe_crp')
        param.numClasses = 100;
        param.lowDim = 300;
        param.featureDim = 9216;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 10; % mini-batch size

        param.knn_const = 2; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.01; % large margin for structure preserving
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 1000; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 10;
    elseif strcmp(method, 'lmspe_le')

    elseif strcmp(method, 'lme_new')
        
    end
end