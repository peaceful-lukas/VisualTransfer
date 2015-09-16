load('/Users/lukas/Desktop/lme_new_pascal3d_pascal_7731.mat')

param = result{1};
W = result{2};
U = result{3};

[coeff, pca_score, latent] = pca(U');  
U_proj = pca_score(:, 1:3)';

protoStartIdx = [0 cumsum(param.numPrototypes)];

figure;
hold on;
for i=1:param.numClasses
    scatter3(U_proj(1, protoStartIdx(i)+1:protoStartIdx(i+1)), U_proj(2, protoStartIdx(i)+1:protoStartIdx(i+1)), U_proj(3, protoStartIdx(i)+1:protoStartIdx(i+1)));
    pause;
end
hold off;
