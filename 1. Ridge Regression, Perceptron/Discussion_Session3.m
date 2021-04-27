%% 1D OLS Example
clear all;
% m: #of data points
% d: #of feature
% w_true: true weights
% w_ols: Ordinary Least Square solution

m = 100;
d = 1;
w_true = 2*rand(d+1,1);
X = [ones(m,1),100*rand(m,d)];
y = X*w_true + 20*rand(m,1);

x_cont = [min(X(:,2)):0.1:max(X(:,2))]';
x_cont = [ones(length(x_cont),1),x_cont];

w_ols = inv(X'*X)*X'*y;


figure(1)
p1 = plot(X(:,2),y,'x','LineWidth',2,'Color',[0,0,1]);
hold on;
p2 = plot(x_cont(:,2),x_cont*w_ols,'Color',[1,0,0]);
title('Ordinary Least Squares','FontSize',20);
xlabel('x','FontSize',20);
ylabel('y','FontSize',20);
legend([p1,p2],{'x_i','LS Solution'},'FontSize',14);

%% d-D OLS example 
% m > d
clear all;
m = 100;
d = 10;

w_true = 2*rand(d+1,1);
X = [ones(m,1),100*rand(m,d)];
y = X*w_true + 0*rand(m,1);

[U,S,V] = svd(X);
U = U(:,[1:11]);
S = diag(S);

w_ols = (X'*X)^-1*X'*y;
w_ols2 = V*diag(1./S)*U'*y;

stem(w_ols,'LineWidth',2);
hold on;
stem(w_ols2,'LineWidth',2);
(w_ols-w_ols2)'*(w_ols-w_ols2)
%% d-D OLS example 
% m < d
clear all;
m = 10;
d = 100;

w_true = 2*rand(d+1,1);
X = [ones(m,1),100*rand(m,d)];
y = X*w_true + 10*rand(m,1);

[U,S,V] = svd(X);
V = V(:,[1:10]);
S = diag(S);

w_ols = pinv(X'*X)*X'*y;
w_ols2 = V*diag(1./S)*U'*y;

stem(w_ols,'LineWidth',2);
hold on;
stem(w_ols2,'LineWidth',2);
(w_ols-w_ols2)'*(w_ols-w_ols2)
%% Linearly Dependent Column
%Truncated SVD
clear all;
m = 100;
d = 10;

w_true = 2*rand(d+2,1);
X = [ones(m,1),100*rand(m,d)];
X = [X,2*X(:,1)+3*X(:,3)];
y = X*w_true + 10*rand(m,1);

[U,S,V] = svd(X);
U = U(:,[1:12]);
S = diag(S);

figure;
plot(eig(X'*X),'LineWidth',2);
title('\lambda_i of X^T X','FontSize',20)

truncation_index = 11;
U = U(:,[1:truncation_index]);
V = V(:,[1:truncation_index]);
S = S([1:truncation_index]);


w_ols = pinv(X'*X)*X'*y;
w_ols2 = V*diag(1./S)*U'*y;

figure;
stem(w_ols,'LineWidth',2);
hold on;
stem(w_ols2,'LineWidth',2);
(w_ols-w_ols2)'*(w_ols-w_ols2)
%% Polynomial Regression
clear all;
% m: #of data points
% d: #of feature
% w_true: true weights
% w_ols: Ordinary Least Square solution

m = 100;
d = 5;
w_true = [0.4,-0.3,0.5,-0.25,-1,0.6]';
x = 2*rand(m,1);
X = [ones(m,1)];
for i = 1:d
    X = [X,x.^i];
end
y = X*w_true + 1/2*rand(m,1);

d = 10;
X = [ones(m,1)];
for i = 1:d
    X = [X,x.^i];
end

w_ols = inv(X'*X)*X'*y;
w_ols2 = inv(X'*X + 1*eye(11))*X'*y;
w_ols3 = inv(X'*X + 10*eye(11))*X'*y;
w_ols4 = inv(X'*X + 100*eye(11))*X'*y;

x_cont = [min(X(:,2)):0.1:max(X(:,2))]';
X_cont = ones(length(x_cont),1);
for i = 1:d
    X_cont = [X_cont,x_cont.^i];
end


figure(1)
p1 = plot(X(:,2),y,'x','LineWidth',2,'Color',[0,0,1]);
hold on;
p2 = plot(x_cont,X_cont*w_ols);
p3 = plot(x_cont,X_cont*w_ols2);
p4 = plot(x_cont,X_cont*w_ols3);
p5 = plot(x_cont,X_cont*w_ols4);
title('Ordinary Least Squares','FontSize',20);
xlabel('x','FontSize',20);
ylabel('y','FontSize',20);
legend([p1,p2,p3,p4,p5],{'x_i','LS Solution','\lambda = 1','\lambda = 10'...
   ,'\lambda = 100' },'FontSize',14);

w_ols = inv(X'*X)*X'*y;