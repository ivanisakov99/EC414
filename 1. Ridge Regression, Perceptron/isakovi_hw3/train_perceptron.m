function [w, b, average_w, average_b] = train_perceptron(x,y)

[m,n] = size(x);
w = zeros(n, 1);
b=0;

for i=1:m
    
    xi = x(i,:);
    yi = y(i);
    
    ytilda = xi*w + b;
    
    if ytilda >= 0
        ytilda = 1; 
    else
        ytilda = -1;
    end
    
    if y(i) ~= ytilda
        w = w' + yi*x(i,:);
        w = w';
    end
      
end

average_w = w/m;
average_b = b/m;