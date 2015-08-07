function saveResult(method, dataset, accuracy, result)
    save_fname = sprintf('/home/twkim/exp_results/%s_%s_%.4f.mat', method, dataset, accuracy);
    save(save_fname, 'result');
end