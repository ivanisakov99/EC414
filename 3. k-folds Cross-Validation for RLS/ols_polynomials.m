close all
clear
X=[-3; 0; 1; -7; 5];
y=[24; -7; 4; 46; 36];
Xtest=[-7:0.01:5]';
plot(X,y,'x')
hold on
plot(Xtest, Xtest.^2,'k')

%Linear predictor
tildeX=[ones(5,1), X];
tildew=pinv(tildeX'*tildeX)*tildeX'*y;
b=tildew(1);
w=tildew(2:end);
plot(X,X*w+b)

size(tildeX)

% quadratic predictor
tildeX2=[ones(5,1), X, X.^2];
tildew2=pinv(tildeX2'*tildeX2)*tildeX2'*y;
b2=tildew2(1);
w2=tildew2(2:end);
plot(Xtest,Xtest.^2*w2(2)+Xtest*w2(1)+b2,'b')

size(tildeX2)

% Polynomial predictor, degree 5
tildeX3=[ones(5,1), X, X.^2, X.^3, X.^4, X.^5];
tildew3=pinv(tildeX3'*tildeX3)*tildeX3'*y;
b3=tildew3(1);
w3=tildew3(2:end);
plot(Xtest, Xtest.^5*w3(5) + Xtest.^4*w3(4) + Xtest.^3*w3(3)+ Xtest.^2*w3(2) + Xtest*w3(1) + b3,'g')

size(tildeX3)

xlabel('x')
ylabel('y')
legend('Training points', 'Best predition function', ' Linear Fit', 'Quadratic Fit', 'Poly Fit, degree 5')
