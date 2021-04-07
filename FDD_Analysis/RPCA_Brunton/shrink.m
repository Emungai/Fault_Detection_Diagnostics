function out = shrink(X,tau)
%Brunton's book chapter 3
    out = sign(X).*max(abs(X)-tau,0);
end