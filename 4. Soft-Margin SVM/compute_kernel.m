function [K] = compute_kernel(X,gamma)
normX = full(sum(X.^2,2));
K = repmat(normX,1,size(X,1)) + repmat(normX',size(X,1),1) - 2*full(X*X');
K=exp(-K*gamma);
end

