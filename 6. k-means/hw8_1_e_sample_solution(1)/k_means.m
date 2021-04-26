function [c,obj,y] = k_means(X, c0, T, dim)
% The optional parameter dim indicates the form of feature vectors in X. 
% dim = 1 indicates that features are column vectors; all other values of
% dim will lead to the default execution of the algorithm, which treats 
% rows of X as feature vectors. dim = 1 seems to work faster on MNIST data
% Reference used: 
% https://statinfer.wordpress.com/2011/12/12/efficient-matlab-ii-kmeans-clustering-algorithm/
c=c0;
if nargin == 3 || dim ~= 1
    k=size(c0,1);
    n=size(X,1);
    % Fake initial assignment
    y=zeros(1,n);
    for iter=1:T
        % calculate matrix of squared distances between centers and points
        norm_c = sum(c.*c,2);
        similarity = c*X' - 0.5*norm_c;
        % assign to centers
        old_y=y;
        [max_sim,y]=max(similarity);
        if old_y==y
            break;
        end
        % update centers
        clusterSizes = accumarray(y',1); 
        E = sparse(y,1:n,1,k,n,n); % turning labels into an indicator matrix
        filter = clusterSizes~=0;
        tmp = E(filter,:)*X;
        % compute center of each non-empty cluster; assignment by index
        % also conveniently preserves the type of c, so full vectors will
        % remain full
        c(filter,:) = tmp./clusterSizes(filter,:); 
    end
    % Squared norms of the samples, transposed
    norm_x = sum(X.*X,2)';
    obj = full(sum(norm_x-2*max_sim));
else
    k=size(c0,2);
    n = size(X,2);
    old_y = 0;
    [max_sim,y] = max((c'*X - 0.5*sum(c.*c,1)'),[],1);
    for iter=1:T
        if y == old_y
            break;
        end
        E = sparse(1:n,y,1,n,k,n);  % transform label into indicator matrix
        clusterSizes = accumarray(y',1);
        filter = clusterSizes~=0;
        c(:,filter) = (X*E(:,filter))./(clusterSizes(filter))';
        old_y = y;
        [max_sim,y] = max((c'*X - 0.5*sum(c.*c,1)'),[],1); % assign samples to the nearest centers
    end
    norm_x = sum(X.*X,1);
    obj = full(sum(norm_x-2*max_sim));
end
fprintf('The algorithm took %d iterations to terminate \n', iter);
end
