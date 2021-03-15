function [lambda5, k_] = skeleton_hw9_2()

%% Load AT&T Cambridge, Face images data set
    img_size = [112,92];   % image size (rows,columns)
    % Load the ATT Face data set using load_faces()
    X = load_faces();
    
    %% Compute mean face and the Auto Covariance Matrix
    % compute X_tilde
    mu = mean(X);
    X_tilde = X - mu;
    
    % Compute covariance using X_tilde
    [m, ~] = size(X);
    covX = (1/m) * (X_tilde' * X_tilde);
    covX = (covX' + covX)/2;
    
    %% Find Eigen Value Decomposition of auto covariance matrix
    [V, D] = eig(covX);
    
    %% Sort eigen values and corresponding eigen vectors and obtain U, Lambda
    [A, idx_sort] = sort(diag(D), 'descend');
    Lambda = diag(A);
    U = V(:, idx_sort);
    
    %% Find principle components: Y
    Y = U(1:400, :)' * X_tilde;
    
%% a) Visualize loaded images and mean face
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image 120 in the dataset
    % practise using subplots for later parts
    subplot(1,2,1)
    imshow(uint8(reshape(X(120,:), img_size)))
    title('Image #120')
    
    % Visualize the Average face image
    subplot(1,2,2)
    imshow(uint8(reshape(mu, img_size)))
    title('Mean image')
    print -dpng Q9_2a.png
    
    
%% b) Analysing computed eigen values
    warning('off')
    
    % Report first 5 eigen values
    % lambda5 = ?; 
    lambda5 = A(1:5);
    
    % Plot trends in Eigen values and k
    k = 1:450;
    figure(2)
    sgtitle('Eigen Value trends w.r.t k')

    % Plot the eigen values w.r.t to k
    subplot(1,2,1)
    
    plot(A(k))
    title('Lambda as a function of k')
    
    % Plot running sum of eigen vals over total sum of eigen values w.r.t k
    % Compute eigen fractions
    for i = 1 : 450
        rho(i) = (sum(A(1:i))) / (sum(A));
    end
    
    subplot(1,2,2)
    plot(rho)
    title('Fraction of variance explained')
    print -dpng Q9_2b.png
    rho = round(rho, 2);
    
    % find & report k for which Eig fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    % k_ = ?; 
    for i = 1:5
        k_(i) = find(rho == ef(i), 1);
    end
    
%% c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = X(test_img_idx,:);    
    % Computing eigen face coefficients
    K = [0,1,2,k_,400];
    j = 1;
    X_hat = zeros(8, 10304);
    for i = 2:9
        s = K(i);
        X_hat(j, :) = (test_img - mu) * U(:, 1:s) * U(:, 1:s)' + mu;
        j = j + 1;
    end
    
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to mean face

    % plot the resultant images from progress of adding eigen faces to the 
    % mean face in a single figure using subplots.
    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    
    subplot(2,5,1)
    imshow(uint8(reshape(test_img, img_size)))
    title('Original image')
    
    subplot(2,5,2)
    imshow(uint8(reshape(mu, img_size)))
    title('Mean image')
    
    subplot(2,5,3)
    imshow(uint8(reshape(X_hat(1, :), img_size)))
    title('k = 1')
    
    subplot(2,5,4)
    imshow(uint8(reshape(X_hat(2, :), img_size)))
    title('k = 2')
    
    subplot(2,5,5)
    imshow(uint8(reshape(X_hat(3, :), img_size)))
    title('k = 6')
    
    subplot(2,5,6)
    imshow(uint8(reshape(X_hat(4, :), img_size)))
    title('k = 29')
    
    subplot(2,5,7)
    imshow(uint8(reshape(X_hat(5, :), img_size)))
    title('k = 105')
    
    subplot(2,5,8)
    imshow(uint8(reshape(X_hat(6, :), img_size)))
    title('k = 179')
    
    subplot(2,5,9)
    imshow(uint8(reshape(X_hat(7, :), img_size)))
    title('k = 300')
    
    subplot(2,5,10)
    imshow(uint8(reshape(X_hat(8, :), img_size)))
    title('k = 400')
    
    print -dpng Q9_2c.png
   

end
