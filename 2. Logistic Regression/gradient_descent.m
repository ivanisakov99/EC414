function [x,obj] = gradient_descent(x0,A,T,eta)
% x0 : Initial x
% A : Matrix defined in objective function
% T: Number of iterations
% eta: Learning rate 
    obj = zeros(T,1); %Allocate space for obj (faster)
    x = x0;
    for i = 1:T % For loop for T iterations
        obj(i) = x'*A*x; %Objective function at i'th iteration
        gradient = (A+A')*x; % Gradient is (A + A^T)x
        x = x - eta*gradient; % Update x
    end
end