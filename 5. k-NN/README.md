# <ins>Problem: ???</ins>

In this problem, we will implement k-NN and use it on the cats vs dogs dataset (see Lecture 12). Warning: the training/validation/test split in the code is random, so you’ll get different results every time you run it.


1. As said in class, k-NN does not require training. So, implement directly the k-NN prediction algorithm we saw in class in a Matlab function with prototype:

        [yhat] = predict_knn(X, y, Xtest, k)

    where yhat is a column vector of predictions for the n testing samples, X is the matrix of the training samples ∈ Rm×d, y is the vector of labels ∈ Rm, Xtest is the matrix of the testing samples ∈ Rn×d, and k the number of neighbors to consider. There is no skeleton code. Hint: to take the majority vote of n numbers in {−1, 1} it is enough to sum them and take the sign. Also, sort in Matlab sorts vector of numbers.

2. Let’s now use the function in the point above to select the best k using a validation set. You decide a good range of value of k to try. Plot the 0/1 loss on the validation set with respect k.

3. Test on the test set using the best k found using the validation set and report the 0/1 error.
