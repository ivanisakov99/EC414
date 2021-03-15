function [w,b,obj] = train_logistic_regression_gd(X,y,eta,T,w0,b0)
[m,n] = size(X);
wt = [b0; w0];
Xt = [ones(m,1), X];
F = zeros(T,1); 

for i = 1:T
    F(i) = sum(log(1 + exp(-y.*(Xt*wt))));

    grad = -sum((y.*Xt)./(1 + exp(y.*(Xt*wt))))';

    wt = wt - eta * grad;
end
b = wt(1);

w = wt(2:end);

obj = F;

end