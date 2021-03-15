function [alpha] = train_kernel_perceptron(X,y,gamma,epochs)

[m,d]=size(X);

K=compute_kernel(X,gamma);
alpha=zeros(m,1);

for i=1:epochs
    rand_idx=randperm(m);
    mistakes=0;
    for l=1:m
        j=rand_idx(l);
    
        if y(j)*alpha'*K(:,j)<=0
            alpha(j)=alpha(j)+y(j);
            mistakes=mistakes+1;
        end
    end
    fprintf('Epoch %d, Mistakes: %d\n', i, mistakes);
    if mistakes==0
        break
    end
end

end

