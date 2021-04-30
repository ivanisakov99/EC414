%% EC414 Introduction to Machine Learning: Discussion Session 4 
% Unay Dorken Gallastegi

clear all;
close all;

%% Some useful vectors

X = zeros(3,2) % Matrix of 0s with size 3x2
X = ones(4,3) % Matrix of 1s with size 4x3
X = rand(5,2) % iid Guassian (zero mean, unit variance) entries with size 5x2

%% Creating a matrix from scalars, vectors or matrices

x = [2, 3, 5, 6] % Use ',' to add a column
x = [2; 3; 5; 6] %Use ';' to add a row

% Add a column/row to a matrix with ','/';'
A = [x, ones(4,1), rand(4,1), zeros(4,1)]
A = [x'; ones(1,4); rand(1,4); zeros(1,4)]

A = [A, A] % Concatenate matrices


% Use length(vector), size(matrix) to get the dimensions
[m,n] = size(A)
d = length(x)

%% Matrix vector indexing
X = rand(5,6)

x = X(3,2) % i = 3 , j = 2 element of matrix X
x = X(:,3) % Third column of X
x = X(2,:) % Second row of X
x = X(:,[1:3]) % First,second and third column of X
x = X([1:4],:) % 1 through 4 rows of X
x = X([1:3],[2:4]) %submatrix of X selecting the first three columns and 2,3,4 rows.

%% Example on how to go through columns/rows of matrx X
X = rand(5,6)

[m,n] = size(X);

% Going through each element of X
for i = 1: m
    for j = 1:n
        X(i,j)
    end
end

% Going through each column of X
for i = 1:n
    X(:,i)
end

%Going through each row of X
for i = 1:m
    X(i,:)
end

%% Vectorized/elementwise operations on Matrix-Matrix

X = rand(3,4);

% Elementwise power of X
p = 2;
A = X.^p

% Elementwise multiplication of two matrices
B = A .* X

% Elementwise division of two matrices
B = A ./ X

%% Vectorized/elementwise operations on Vector-Vector
x = [1:4]'
y = [1:4]

z = x .* y' % elementwise multiplication
z = x' .* y % elementwise multiplication
z = x .* y % Gives a matrix where i,j'th element is x(i)*y(j)

%% Vectorized/elementwise operations on Matrix-vector
X = rand(5,6);
a = rand(5,1);
b = rand(1,6);

A = a .* X %elementwise multiplication of columns of X with a
A = b .* X %elementwise multiplication of rows of X with b
%% Some Useful Functions
X = rand(5,6)

A = exp(X) % elementwise exponential function
A = log(X) % elementwise natural logarithm 
A = sum(X,1) % summation on the first dimension
A = sum(X,2) % summation on the second dimension

%% Example: Gradient and objective function for logistic regression evaluated at w0,b0
load adult_train_test.mat;

[m,d] = size(Xtrain); %Assign dimensions of X
b0 = rand(1); %Specify evalution points
w0 = rand(d,1); %Specify evalution points

X_tilde = [ones(m,1),Xtrain]; %Calculate augmented matrices
w_tilde = [b0;w0]; %Calculate augmented matrices

X_tilde*w_tilde; %Calculates w^T*x_i for every i
(-ytrain.*(X_tilde*w_tilde)) %Calculates -y_i*w^T*x_i for every i
exp(-ytrain.*(X_tilde*w_tilde)) %Exponential function computes elementwise
1 + exp(-ytrain.*(X_tilde*w_tilde)) % +1 adds 1 to every element
log(1 + exp(-ytrain.*(X_tilde*w_tilde))) %Log function computes elementwise
sum(log(1 + exp(-ytrain.*(X_tilde*w_tilde)))) %sum over every i \in {1,...,m}

F = sum(log(1 + exp(-ytrain.*(X_tilde*w_tilde)))); %Objective function in one line
gradient = -sum((ytrain.*X_tilde)./(1 + exp(ytrain.*(X_tilde*w_tilde))))'; %Gradient function in one line    

%% Example: Logistic Regression Test after gradient descent
p_y = 1./(1 + exp(-(Xtest*w0 + b0))); % P(y = 1 | X = x)
y_est = sign(p_y - .5); % As 0 < P(y = 1 | X = x) < 1
accuracy = length(find( ytest - y_est == 0))/length(ytest)

%% How to write a custom function Gradient descent example for minimizing x'Ax
% 1) Open a new script
% 2) Define as a function: function [output1,...,outputn] = my_function(input1,...,inputm)

%Example: find x^* = argmin_x x^T*Ax  
%Solution: Use gradient descent where gradient is (A + A^T)x

clear all;
close all;
load A;

x0 = rand(10,1); %Initialize x

T = 1000; %Specify number of iterations
eta = 1e-4; % Specify learning rate
[x,obj] = gradient_descent(x0,A,T,eta); %Call the custom function gradient_descent

plot(obj,'LineWidth',2,'Color',[0,0,1]);
xlabel('Iteration','FontSize',20);
ylabel('Objective Function','FontSize',20);
