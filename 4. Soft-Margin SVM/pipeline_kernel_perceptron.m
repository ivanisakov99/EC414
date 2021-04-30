%% Example
clear
close all

[X,y] = read_data; %Load Data

[m d] = size(X);
ntrain = round(m*0.60);
nval = round(m*0.20);
order = randperm(m);
Xtrain = X(order(1:ntrain),:);
ytrain = y(order(1:ntrain));
Xval = X(order(ntrain+1:ntrain+nval),:);
yval = y(order(ntrain+1:ntrain+nval));
Xtest = X(order(ntrain+nval+1:end),:);
ytest = y(order(ntrain+nval+1:end));

gamma_values=10.^(-5:1:0);
%gamma_values=10.^(-10:1:-5);

for j=1:numel(gamma_values)
    gamma=gamma_values(j)
    alpha=train_kernel_perceptron(Xtrain,ytrain,gamma,20);

    mistakes(j)=0;
    for i=1:size(Xtest,1)
        yhat=test_kernel_predictor(Xtrain, alpha,Xval(i,:),gamma);
        mistakes(j)=mistakes(j)+(sign(yhat)~=yval(i));
    end
end

plot(mistakes/numel(yval))

[mn,idx_mn]=min(mistakes);
gamma_star=gamma_values(idx_mn);
alpha=train_kernel_perceptron([Xtrain;Xval],[ytrain;yval],gamma_star,20);
%alpha=train_kernel_perceptron([Xtrain;Xval],[ytrain;yval],gamma_star,10000);
mistakes=0;
for i=1:size(Xtest,1)
    yhat=test_kernel_predictor([Xtrain;Xval], alpha,Xtest(i,:),gamma);
    mistakes=mistakes+(sign(yhat)~=ytest(i));
end
fprintf('Test Error: %f\n', mistakes/numel(ytest));
