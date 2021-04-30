# <ins>Problem: Gradient Descent for Logistic regression</ins>

In this problem, we will implement the Gradient Descent (GD) Algorithm to learn the parameters of a
logistic regression classifier. We will use again the Adult dataset.

//

1. Complete the skeleton of *train_logistic_regression_gd.m* implementing Algorithm 1 in
a function with prototype:

        [w, b, obj] = train_logistic_regression_gd(X, y, eta, T, w1, b1)

    where w ∈ Rd and b ∈ R are the hyperplane vector and the bias found by GD, obj is the vector of objective function values, X is the matrix of the training samples ∈ Rm×d, y is the vector of labels ∈ Rm, eta is the learning rate, T is the maximum number of iterations to run, w1 ∈ Rd is the initialization for w, and b1 ∈ R is the initialization for b. Note: It will be faster if you use Matlab vectorized operations to calculate the gradient, but it is also fine to use for loops.

2. In class, we said that using a random initialization to minimize a convex function should be avoided. Here, we will test this claim. Complete the skeleton code in problem 4 3b.m to use your implementation of logistic regression on the Adult dataset for T = 1000 iterations, η = 1/5000, and 10 random initializations using Gaussian with zero mean and variance 1 on each coordinate of w and on b (You can generate Gaussian noise in Matlab with randn).

3.  Plot the 10 lines of the values of the objective function during training in the point above. What do you observe? Based on these results and what we said in class, why random initialization should be avoided? (To better observe the difference among the lines, use loglog that uses a logarithmic scale on both axes.)

4. Complete the skeleton code in problem 4 3d.m to use your implementation of GD for logistic regression, starting from the zero vector and zero bias, T = 1000 iterations, η = 1/5000, and test the obtained solution on the test set. Count the number of prediction mistakes on the test set, predicting for each sample the class with the highest probability.

5. If the learning rate is too big, you should get a lot of “Inf”. Verify it yourself using for example η = 0.1. Propose a way to fix this issue. Hint: First, locate the cause of the problem, then try to understand why we get this problem numerically. Finally, think how to fix the problem. You can propose approximations.
