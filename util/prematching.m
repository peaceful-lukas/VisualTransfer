function score = prematching(L1, L2)

minNumEl = min(size(L1, 1), size(L2, 2));

[~, d1] = eig(L1, 'vector');
d1 = d1./max(d1);
d1 = sort(d1, 'descend');
d1 = d1(1:minNumEl); % the first element always has the value 1

[~, d2] = eig(L2, 'vector');
d2 = d2./max(d2);
d2 = sort(d2, 'descend');
d2 = d2(1:minNumEl);

score = exp(-sum(abs(d1 - d2)));

