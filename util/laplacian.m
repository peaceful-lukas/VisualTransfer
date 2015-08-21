function L = laplacian(A, normalized)

if normalized
    invSqrtD = pinv(sqrt(diag(sum(A, 1))));
    L = eye(size(A)) - invSqrtD*A*invSqrtD;
else
    D = diag(sum(A, 1));
    L = D - A;
end
