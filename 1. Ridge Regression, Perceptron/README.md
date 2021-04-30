# <ins>Problem : Ridge Regression</ins>
Here, you will code a Matlab (or Octave) function that implements the Least Square (LS) algorithm seen in class and a generalization called Regularized Least Squares (RLS), or <ins>Ridge
Regression</ins>. We will apply RLS to a real-world 8-dimensional (8 features) prostate cancer dataset contained in the file *prostateStnd.mat*. In this dataset, 8 medically relevant features
named lcavol, lweight, age, lbph, svi, lcp, gleason, and pgg45 are used to estimate lpsa (log prostate specific antigen). The training and test data are provided as *(xtrain, ytrain)* and
*(xtest, ytest)* respectively. The first 8 features correspond to the first 8 entries of names. The ninth entry of names (the last one) is the label to be predicted whose values are in 
*(ytest, ytrain)*.

Using the notation defined above, the LS problem is:

//

The RLS problem is:

// 

where λ is the <ins>regularization parameter</ins>. We still don’t know what a regularizer is and why we should use it. Yet, here we will try to gather some practical intuition on it. You can see that LS is nothing else than RLS with λ = 0. Hence, we can just implement RLS.

1. As a first step, write Matlab code to normalize the training dataset so that post-normalization, each of the 8 features and the label in the normalized training dataset has zero mean and unit variance. This requires determining a pair of offset and scaling parameters, one pair for each feature and one pair for the label. These parameters must be computed only from the training dataset, but they must be applied to both the training and test datasets, i.e., we normalize both the training and test data, but we are only allowed to normalize the test data using parameters derived from the training data. In other words we must apply identical operations to training and test data. Only the training data will be actually normalized by the operation. If the test data is statistically similar to the training data, it too will be approximately normalized. The prototype must be:
    
        [XtrainNormalized, XtestNormalized] = normalize_data(Xtrain, Xtest)

2.  

//

3. From the above, implement RLS with prototype:

        [w, b] = train_rls(X, y, lambda)

    The code must be robust to the case that the matrix C is not invertible.

4. Usethenormalizeddatatotrainaridgeregressionmodelforeachofthefollowingvaluesofthe regularization penalty parameter λ: {e−5, e−4, e−3, ..., e10}. In a single figure, plot the ridge regression coefficient of each feature (8 in total) as a function of ln λ (8 curves in total) for ln λ ranging from -5 to 10 in steps of 1. Use suitable colors and/or markers to distinguish between the 8 curves and label them appropriately in a legend. Discuss what happens to the coefficients as λ becomes larger. (No code to submit here, just plots and your comments)

5. In another figure, plot the mean-squared-error (MSE) of both the training and the test data as a function of ln λ. Discuss your observations. (No code to submit here, just plots and your comments)

# <ins>Problem : Perceptron</ins>

Here, we will implement and test the Perceptron algorithm on a real-world binary classification dataset.

//

1. Consider the alternative Perceptron updates w ̃ ← w ̃ + ηyi x ̃ i with learning rate η > 0. The Perceptron algorithm is the special case η = 1. Consider this alternative Perceptron and the standard Perceptron on the same sequence of training samples. Prove that these two algorithm with make exactly the same mistakes regardless of the value of η > 0.

2. Implement the Perceptron algorithm in Algorithm 1. The prototype of the function must be:

        [w, b, average_w, average_b] = train_perceptron(X, y)

    where w and b are the last solutions, while average_w and average_b are the averaged solutions

3. Now, let’s test the Perceptron algorithm on the training data of Adult UCI dataset, where the binary classification task is to determine whether a person makes over 50K a year. The Perceptron has to do only one pass over the data. The test_adult will run your code and report the performance of the last solution and of the average solution on the test set, shuffling the training data and repeating the above 10 times. What do you observe? (No code to submit here, just numbers and your comments)