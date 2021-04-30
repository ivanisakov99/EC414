function val = Kernel(X1, X2, gamma)
temp = reshape(X1, length(X1(:,1)), 1, length(X1(1,:)));
temp2 = reshape(X2, 1, length(X2(:,1)), length(X2(1,:)));
temp = temp2 - temp;
val = exp(-gamma*sum(temp.^2,3));
end