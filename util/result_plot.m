figure;
hold on;


bar([lme_sp_accuracy lmspe_le_accuracy lmspe_accuracy], 'grouped');
axis([0 21 0 1]);
category_labels = {'airplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'table'; 'dog';
                    'horse'; 'motorbike'; 'person'; 'plant'; 'sheep'; 'sofa'; 'train'; 'monitor'; 'cow'; 'chair'};

xticklabel_rotate([1:20],45, category_labels);


xlabel('categories');
ylabel('accuracy');
legend('LME', 'LMSPE-LE', 'LMSPE(ours)');
hold off;








[~, classified] = max(DS.T'*W'*U, [], 2);
classified = ceil(classified/param.numPrototypes);
accuracy = numel(find(DS.TL == classified))/numel(DS.TL);

accuracies = zeros(20, 1);
numClassData = zeros(20, 1);
for n=1:20
    idx = find(DS.TL == n);
    class_accuracy = numel(find(classified(idx) == n))/numel(idx);
    accuracies(n) = class_accuracy;
    numClassData(n) = numel(idx);
end