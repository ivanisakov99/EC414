# <ins>Problem: k-means Implementation</ins>

In this problem we will implement k-means clustering and explore the impact of initialization and number of clusters on one synthetic and one real-world dataset. We will also explore a dataset where k-means will fail to produce meaningful clusters. You are provided skeleton code to assist you in implementing this clustering method.

1. Implement the k-means algorithm we saw in class in Matlab with prototype:

        [c,obj,y] = k_means_clustering(X, c0, T)

    where c are the returned centers ∈ Rk×d, obj is the final value of the objective function, y are the inferred “labels”, X is the matrix of the training samples ∈ Rm×d, c0 are the initial centers ∈ Rk×d, T is the maximum number of iterations. Note that there is no need to pass k because the algorithm can infer it from the size of c0.

2. Generate3two-dimensionalGaussianclustersofdatapoints
having the following mean vectors and covariance matrices: μ1 = [2,2]⊤, μ2 = [−2,2]⊤, μ3 =
[0,−3.25]⊤,andΣ1 =0.02·I2,Σ2 =0.05·I2,Σ3 =0.07·I2,whereI2 isthe2×2identitymatrix.You
can use mvnrnd to generate multivariate Gaussian noise. Let each data cluster have 50 points. Plot
the generated Gaussian data. Color the data points in the 1st, 2nd, and 3rd clusters with red, green,
and blue colors, respectively. You can use gscatter to easily plot the points in different colors.
Use your implementation of k-means on this dataset with k = 3 and the following initialization:
cinitial = [3, 3]⊤, cinitial = [−4, −1]⊤, cinitial = [2, −4]⊤. and the maximum number of iterations equal 123
to 10. Plot the clusters produced by your k-means algorithm, plotting the points of each cluster with a different color.

3. Using the same synthetic training dataset from part (b), re-
run your k-means algorithm implementation for k = 3 using the following (different) initialization:
cinitial = [−.14, 2.61]⊤, cinitial = [3.15, −0.84]⊤, cinitial = [−3.28, −1.58]⊤. Create a new plot of the 123
resulting clusters. Discuss what you observe.

4.  To reduce the possibility selecting an initialization which results in a “bad” clustering, the k-means algorithm is typically run multiple times using dif- ferent random initializations. The best clustering result, i.e., the one having the smallest objective function is saved and used as the final output. Run your implementation of the k-means algorithm on the same synthetic training dataset from part (b) for 10 different random initializations. Report the objective function for each of the 10 trials. Identify the trial which yields the smallest objective function value. Report its objective function value and create a plot of the clustering produced by it.

5. Here we examine a real-world digits, MNIST. The inputs are 28 x 28 images of digits flattened to vectors of size 784. Ignore the labels and apply your im- plementation of the k-means algorithm over the training data with k = 10 selecting the best of 10 different random initializations as the final output. Reshape the centers as images and plot the images corresponding to the 10 centers. Hint: you can reshape and display the first training sample with the following command imagesc(reshape(Xtr(1,:),28,28)’). Also, if your implementation of k-means is too slow, take a subset of the training set.

6. Here we examine the performance of the k-means algorithm on a dataset composed of 3 concentric rings. Use sample circle.m to generate a dataset with 3 concentric ring clusters and 500 points for each cluster. Plot the dataset. Apply your implementation of the k-means algorithm on this dataset using k = 3 and choosing the best of 10 different random initializations. Create a plot of the best clustered results. Discuss what you observe.
