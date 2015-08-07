function S = conSimMat(x1, x2)
% Compute similarity matrix
%
% Usage
%   x1 = rand(3, 10); 
%   x2 = rand(3, 5);
%   S = conSimMat(x1, x2);
%
% Input
%   x1      - columnwise concatenation of sample instances, dim x n1
%   x2      - columnwise concatenation of sample instances, dim x n2
% 
% Output
%   S       - similarity matrix, n1 x n2
%
% History
%   create - Taewoo Kim (twkim@unist.ac.kr), 07-28-2015
%
%
% Inspired by Feng Zhou's Matlab Library

if nargin == 1
    S = x1'*x1;
else
    S = x1'*x2;
end