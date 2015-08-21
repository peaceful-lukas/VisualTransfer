function saveResult(method, dataset, accuracy, result)
    save_fname = sprintf('/v9/exp_results/%s/%s_%s_%d.mat', dataset, method, dataset, round(10000*accuracy));
    save(save_fname, 'result');
end
