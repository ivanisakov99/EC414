function [w, b] = train_rls(Xtrain, ytrain, lambda)
    [m,n] = size(Xtrain)
    A = [ones(m,1),Xtrain];
    C = A'*A;
    I = eye(n);
    I = [zeros(n,1),I];
    I = [zeros(1,n+1);I];
    B = pinv(C + lambda*I)*A'*ytrain;
    b = B(1,:);
    w = B(2:end,:);
end