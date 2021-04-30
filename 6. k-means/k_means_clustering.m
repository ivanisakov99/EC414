function [c, obj, y] = k_means_clustering(X, c0, T)
[m, d] = size(X);
[k, d] = size(c0);

y_hat = 0;
distance = zeros(m, k);
for j = 1:T   
    
    for l = 1:k
        distance(:,l) = sum((X - c0(l,:)).^2, 2);
    end

    
    [objf, y_hat] = min(distance, [], 2);
    
    obj = sum(objf);
    
    for i = 1:k
        index = find(y_hat == i);
        c0(i, :) = mean(X(index, :));
    end
end

c = c0;
y = y_hat;

% k=size(c0,1);
% [m,d]=size(X);
% 
% c=c0;
% 
% % Norms of the samples
% norm_x = full(sum(X.^2,2));
% 
% % Fake initial assignment
% y=ones(1,m);
% 
% for iter=1:T
%     %iter
%     
%     old_y=y;
%     % calculate matrix of squared distances between centers and points
%     norm_c = full(sum(c.^2,2));
%     dists = repmat(norm_x',k,1) + repmat(norm_c,1,m) - 2*full(c*X');
%     
%     % assign to centers
%     [mn,y]=min(dists);
%     obj=sum(mn);
%     if old_y==y
%         break;
%     end
% 
%     % update centers
%     for i=1:k
%         if numel(find(y==i))~=0    
%             c(i,:)=mean(X(y==i,:));
%         end
%     end  
% end





end

