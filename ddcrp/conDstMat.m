function D = conDstMat(x1, x2)
% Compute distance matrix
%
% Usage
%   x1 = rand(3, 10); 
%   x2 = rand(3, 5);
%   D = conDstMat(x1, x2);
%
% Input
%   x1      - columnwise concatenation of sample instances, dim x n1
%   x2      - columnwise concatenation of sample instances, dim x n2
% 
% Output
%   D       - distance matrix, n1 x n2
%
% History
%   create - Taewoo Kim (twkim@unist.ac.kr), 07-16-2015
%
%
% Inspired by Feng Zhou's Matlab Library

if nargin == 1
    D_vec = pdist(x1');
    D = squareform(D_vec);

else
    % matrix dimension
    n1 = size(x1, 2);
    n2 = size(x2, 2);

    % compute Euclidean distance matrix
    D = zeros(n1, n2);
    for i=1:n2
        D(:, i) = sum(bsxfun(@minus, x1, x2(:, i)).^2, 1)';
    end

    D = sqrt(D);
end