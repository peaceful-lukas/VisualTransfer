function param = getParam(method, dataset)
    param = {};

    switch dataset
        case 'pascal'
            param = getPascalParam(method);
        
        case 'pascal3d'
            param = getPascal3dParam(method);
        
        case 'awa'
            param = getAwaParam(method);
        
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

    end

end


function param = getPascal3dParam(method)
    
    if strcmp(method, 'lme_sp')
        param.numClasses = 12;
        param.lowDim = 30;
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
        param.sp_lm = 0.1; % large margin for structure preserving
        param.sim_bound = 30; % lower bound for the similarities in the pulling term
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 1000; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
        param.bal_c = 1;
        param.bal_sp = 10;
        param.bal_b = 0.0001;

    elseif strcmp(method, 'lmspe_le')

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

        % SAMPLE
        param.numClasses = 50;
        param.maxIterW = 1000;
        param.maxIterU = 1000;
        param.maxAlter = 50;
        param.batchSize = 10; % mini-batch size
        param.lowDim = 100;
        param.featureDim = 9216;

        param.knn_const = 3; % constant for constructing k-nn graph.
        param.c_lm = 10; % large margin for classification
        param.sp_lm = 0.1; % large margin for structure preserving
        param.sim_bound = 30; % lower bound for the similarities in the pulling term
        param.lambda_W = 100000; % regularizer coefficient
        param.lambda_U = 100; % regularizer coefficient
        param.alpha = 5; % softmax parameter.
        param.lr_W = 0.00001; % learning rate for W
        param.lr_U = 0.00001; % learning rate for U
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
    end

end

