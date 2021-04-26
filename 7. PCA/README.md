# <ins>Problem: PCA of Images</ins>

In this problem we will perform Principal Component Analysis (PCA) on the AT&T Face Dataset. We provided a skeleton code, but you don’t have to submit any code for this problem, just figures and results.

**AT&T Face Dataset:** This dataset contains 400 images (10 images of 40 diffferent people) of size 112 × 92 (10304 pixels) in the Portable Grayscale Map (PGM) format. We have provided you this dataset as att-database-of-faces.zip and a helper function load faces.m to load the dataset into your MAT- LAB workspace as a matrix of “vectorized” images of dimensions 400 × 10304. Unzip the dataset and place the resulting folder named att-database-of-faces in the same directory as your solution file to use load faces.m.

1. Use the provided load faces.m function and load the face-dataset into your workspace. Compute the mean face image, that is the mean of all the 400 face image vectors. Use subplot and imshow(uint8(reshape(img vector, img size))) to create a figure containing two images: (1) image #120 in the dataset, and (2) the mean face of the dataset. Does the mean face resemble a human face or is it a smooth shapeless “blob”?

2. Subtract the vectorized mean face image μˆx from all the vectorized face images and compute the empirical covariance matrix Sˆx of the mean-centered vectorized face images. Perform an Eigenvalue Decomposition of this covariance matrix using the inbuilt MATLAB fucntion eig. Arrange the eigenvectors u1, u2, . . . in the order of non-increasing eigenvalues λ1 ≥ λ2 ≥ . . ., into an orthonormal matrix U = [u1, u2, . . .]. Let Λ be the corresponding diagonal matrix of eigenvalues.

    * Report the values of the first five eigenvalues.
    *  Plot λk as a function of k, for k = 1, 2, . . . , 450. Comment on the observed trends in your
    report. Explain why λk = 0 for all k > 400.
    * Compute and plot (as a function of k) the values of the so-called “fraction of variance
    explained” by the top k principal components:

    //

    Round the values of ρk to 2 decimal places. Comment on the observed trends in your report.
    * Find and report the smallest values of k for which ρk ≥ 0.51, 0.75, 0.90, 0.95, and 0.99.

    **Notes:** Use subplots to show plots of λk and ρk in the same figure. The reshaped principal directions of a human face dataset are often referred to as eigenfaces.

3. The first eigenface corresponds to the largest eigen- value and represents a direction that encompasses the largest variance in the training data, the second eigenface corresponds to the second largest eigenvalue and represents a direction which is orthogonal to the first eigenface and encompasses the second largest variance in the training data, and so on. A linear combination of eigenfaces along with a mean face, can be used to reconstruct a given face. Specifically, let x denote the vectorized representation of an image. An approximation to x based on the top k principal components is given by: 

//

Generate a single figure (using subplots) showing the following: the mean face, the k principal component approximation face for image #43 in the face dataset for k = 1, k = 2, all the k values that you found in part (b) sub-part (4), and then show the original image in the same figure too. Add a descriptive title to each image in the figure and a title to the overall figure.
