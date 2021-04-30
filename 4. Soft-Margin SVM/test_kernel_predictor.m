function [yhat] = test_kernel_predictor(X, alpha,xtest,gamma)
for i=1:size(X,1)
    K(1,i)=exp(-gamma*norm(X(i,:)-xtest)^2);
end
yhat=K*alpha;
end

