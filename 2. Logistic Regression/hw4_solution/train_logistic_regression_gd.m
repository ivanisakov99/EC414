function [w,b,obj] = train_logistic_regression_gd(X,y,eta,T,w0,b0)

[m,d]=size(X);

tilde_w=[b0; w0];
tilde_X=[ones(m,1) X];

for i=1:T
    obj(i)=sum(log(1+exp(-y.*(tilde_X*tilde_w))));
    g=-sum(repmat(y./(1+exp(y.*(tilde_X*tilde_w))),1,d+1).*tilde_X)';
    tilde_w = tilde_w - eta *g;
end
b=tilde_w(1);
w=tilde_w(2:end);
