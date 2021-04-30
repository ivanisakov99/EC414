# <ins>Problem: k-folds cross-validation for Regularized Least Square Regression with polynomials</ins>

In this question, you have to implement linear regression from Rd to R using polynomials and choose the best degree of the polynomials using k-folds cross-validation. In other words, for each coordinate i of the input xi, we generate new features (xi2, xi3, . . . , xip) and we append them to the original features. Then, we learn a linear predictor in this space. This corresponds to learn a predictor of the form

//

We also assume that all the coordinates of the input are positive.

1. First, complete the skeleton code of generate_poly_features.m to implement a function that generates a matrix of input samples that contains the polynomials of each feature from the linear term to the polynomial of degree p, with prototype:

        [X_poly] = generate_poly_features(X,p)

    Using our notation, X is m × d, where m is the number of training samples and d is the dimension. X_poly will have the same number of rows and columns equal to d × p. Do not include the term of order 0, that is, the column of 1s.

2. Complete the skeleton code of the function cross_validation_rls.m to implement k-folds cross-validation for regularized least square. The prototype is:

        [validation_loss] = cross_validation_rls(X,y,lambda,k)

    As we have seen in class, in k-folds cross-validation, we divide the training data contained in X and y in k disjoint sets. We assume the training data to be shuffled, so it does not matter how you create the folds. Then, we use one of the k folds as the validation fold and we use the remaining k − 1 to train our RLS predictor with λ =lambda. Evaluate the loss of the trained predictor on the validation fold, repeat the above k times, and return the averages of the mean losses on the validations folds in validation_loss. Note that validation_loss is a scalar. The function to run RLS is also provided in train_rls.m.

3.  In the zip file there is also the “cadata” training/test data in the file “cadata train test.mat”. It is a random train/test split of the Housing dataset from the UCI repository. The task is to pre- dict the median house value from features describing a town. I normalized the features for you, to be in [0, 1]. Complete the skeleton code in problem_5_2c.m to use cross_validation_rls and generate_poly_features to try polynomials up to degree 10 with 8-folds cross-validation and λ = 0.001. The code should record the cross-validation loss for each choice of the degree of the polynomial from 1 to 10 in a vector of dimension 10.

4. Plot the 8-folds validation loss for each degree of the polynomial using the code in the previous point and report the degree of the polynomial that gives the best 8-folds cross-validation loss. Discuss the results: is this what you expected? Does it make sense? (No code to submit here.)

5. Re-train a RLS predictor with the best degree found in the previous point and report its mean square loss error on the test set. (No code to submit here.)
