function val = sumMat(A)
% Sum of all the elements in the matrix
%
% Usage
%   val = sumMat(A);
%
% Input
%   A       - 2-dimensional matrix, n x m
% 
% Output
%   val     - sum of all the elements in the matrix
%
% History
%   create - Taewoo Kim (twkim@unist.ac.kr), 07-16-2015
%
%
% Inspired by Feng Zhou's Matlab Library


val = sum(sum(A));