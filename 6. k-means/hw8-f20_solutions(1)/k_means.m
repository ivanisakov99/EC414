function [c,obj,y] = k_means(X, c0, T)

k=size(c0,1);
[m,d]=size(X);

c=c0;

% Norms of the samples
norm_x = full(sum(X.^2,2));

% Fake initial assignment
y=ones(1,m);

for iter=1:T
    %iter
    
    old_y=y;
    % calculate matrix of squared distances between centers and points
    norm_c = full(sum(c.^2,2));
    dists = repmat(norm_x',k,1) + repmat(norm_c,1,m) - 2*full(c*X');
    
    % assign to centers    
    [mn,y]=min(dists);
    obj=sum(mn);
    if old_y==y
        break;
    end

    % update centers
    for i=1:k
        if numel(find(y==i))~=0    
            c(i,:)=mean(X(y==i,:));
        end
    end  
end

end
