%% Load AT&T Cambridge, Face images data set
img_size = [112,92];   % image size (rows,columns)
X = load_faces();
[n,d] = size(X);

%% Compute mean face and the Auto Covariance Matrix
% compute X_tilde = X - mean(X)
% X_tilde = X - repmat(mean_face, [n,1]); % Alternate
mean_face = mean(X);
X_tilde = X - mean_face;

% Compute covariance using X_tilde
% covX = cov(X); % Alternate
covX = (1/(n-1))*(X_tilde')*(X_tilde);

%% Find Eigen Value Decomposition
% Takes about 340 seconds to finish
[V, D] = eig(covX);

%% Sort eigen values and corresponding eigen vectors
D = diag(D);
[~,idx] = sort(D,'descend');
Lambda = D(idx);
U = V(:,idx);
Y = X_tilde*U;

%% a) Visualize loaded images and mean face
figure(1)
sgtitle('Data Visualization')

% Visualize image 120 in the dataset
subplot(1,2,1)
imshow(uint8(reshape(X(120,:), img_size)))
title('Image 120 in Dataset')

% Visualize the Average face image
subplot(1,2,2)
imshow(uint8(reshape(mean_face, img_size)))
title('Mean Face')

%% b) Analysing computed eigen values
warning('off')

% Report first 5 eigen values
lambda5 = Lambda(1:5);

% Plot trends in Eigen values and k
k = 1:d;
figure(2)
sgtitle('Eigen Value trends w.r.t k')

% Plot the eigen values w.r.t to k
subplot(1,2,1)
plot(k, Lambda, 'r*-')
xlabel('k')
ylabel('Eigen value')
title('Eigen values vs k')

% Plot running sum of eigen vals over total sum of eigen values w.r.t k
numerator = zeros(1,d);
eig_sum = 0;
for i = 1:d
    eig_sum = eig_sum + Lambda(i);
    numerator(1,i) = eig_sum;
end
eig_frac = round(numerator./sum(Lambda), 2);
subplot(1,2,2)
plot(k, eig_frac, 'r*-')
xlabel('k')
ylabel('Sum of k eigen values over sum of all the eigen values')
title('Eigen Fraction vs k')

% find & report k for which Eig fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
ef = [0.51, 0.75, 0.9, 0.95, 0.99];
[~, kidx] = ismember(ef, eig_frac);
k_ = k(kidx);

%% c) Approximating an image using eigen faces
test_img_idx = 43;
test_img = X(test_img_idx,:);
% Computing eigen face coefficients
c_1 = Y(test_img_idx,:);

K = [0,1,2,k_,400,d];
[~,k] = size(K);

figure(3)
sgtitle('Approximating original image by adding eigen faces')
for j = 1:k
    new_img = mean_face;
    % add eigen faces weighted by eigen face coefficients to the mean face
    for i = 1:K(j)
        new_img = new_img + c_1(i)*U(:,i)';
    end

    % Plot formatting
    if j <= k/2
        subplot(3,k/2,j)
        imshow(uint8(reshape(new_img,img_size)))
    else
        subplot(3,k/2,j)
        imshow(uint8(reshape(new_img,img_size)))
    end
    if j == 0
        title('Mean Image')
    else
        title(sprintf('#Eigen Faces = %d', K(j)))
    end
end
% plot the original image
subplot(3,k/2,k+3)
imshow(uint8(reshape(test_img,img_size)))
title('Test image')
