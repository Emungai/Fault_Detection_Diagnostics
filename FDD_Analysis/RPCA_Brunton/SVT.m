function out = SVT(X,tau)
%Brunton's book chapter 3
    [U,S,V] = svd(X,'econ');
    out = U*shrink(S,tau)*V';
end