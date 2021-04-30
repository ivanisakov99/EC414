# <ins>Problem: Soft-Margin Binary Kernel SVM</ins>

In this problem, we will implement and use the Subgradient Descent Algorithm to learn the parameters of the soft-margin Kernel SVM classifier. Let’s start summarizing the derivation we did in class. The following notes are a useful companion to the lecture slides as well.

//

//

//

//

Note that for non-differentiable functions we said that we should take the average of the solutions and use a learning rate of √1 . However, this is an easier problem to solve (because it is 1-strongly convex), so we can
t
use a learning rate that goes to zero faster and we can avoid the averaging.

1. Implement the algorithm in Algorithm 1 in a Matlab function with prototype:

        [alpha, b, obj, zeroOneAverageLoss] = train_ksvm_sd(X, y, T, c, gamma)

    where alpha ∈ Rm and b ∈ R are the solution found by subgradient descent, obj ∈ RT is the vector of objective function values, zeroOneAverageLoss ∈ RT is the vector of the average 0/1 loss on the training samples, X is the matrix of the training samples ∈ Rm×d, y is the vector of labels ∈ Rm, T is the maximum number of iterations to run, c is the trade-off hyperparameter in Eq. (1), gamma is the bandwidth parameter of the Gaussian kernel. There is no skeleton code this time.

2. Create a plot which shows how the objective function F evolves with iteration number t. Is the objective function decreasing? Should it be decreasing? Discuss its behavior.

3. Create a plot which shows how the average 0/1 loss on the training set evolves with iteration number. Is it decreasing? Should it be decreasing? Discuss its behavior.

4. After the subgradient descent algorithm terminates,report the final training error measured with the 0/1 loss. Discuss your observations.

5. (Visualization of decision boundary and margins) Plot on the same graph the following: (1) the training set (use different colors for the two classes) and (2) the decision boundary and the margins of the trained Kernel SVM. Hint: To plot the decision boundary and the margins, you can use the function contour, look for its Matlab help.

