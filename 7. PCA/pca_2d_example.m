%% Example of PCA in 2D
clear all
close all

x1=[-1:0.1:1];
x2=x1.^2;

X=[x1' x2'];
[m,d]=size(X);
y=ones(m,1);
y(abs(x1)<0.5)=-1;

figure
gscatter(x1,x2,y)
grid on
xlabel('x_1')
ylabel('x_2')

figure
gscatter(x1-mean(x1),x2-mean(x2),y)
grid on
xlabel('x_1')
ylabel('x_2')

C=(X-mean(X))'*(X-mean(X));
[V,D]=eig(C);
[~,idx_sort]=sort(diag(D),'descend');
hathatX=(V(:,idx_sort(1))*V(:,idx_sort(1))'*(X-mean(X))')'+mean(X);

figure
gscatter(hathatX(:,1),hathatX(:,2),y)
grid on
xlabel('x_1')
ylabel('x_2')
