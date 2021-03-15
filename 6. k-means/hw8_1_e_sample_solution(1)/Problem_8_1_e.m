% k-means on MNIST
clear
close all

T=70;

load mnist2

[d,m]=size(Xte); % Using Xte because it is conveniently smaller than Xtr 
c = 0; 
obj=inf;
rng(0);
tic
inds0 = find(yte==0);
inds1 = find(yte==1);
inds2 = find(yte==2);
inds3 = find(yte==3);
inds4 = find(yte==4);
inds5 = find(yte==5);
inds6 = find(yte==6);
inds7 = find(yte==7);
inds8 = find(yte==8);
inds9 = find(yte==9);
for i=1:10
    rind = randi(800);
    c0 = zeros(784,10);  
    c0(:,1) = Xte(:,inds0(rind));
    c0(:,2) = Xte(:,inds1(rind));
    c0(:,3) = Xte(:,inds2(rind));
    c0(:,4) = Xte(:,inds3(rind));
    c0(:,5) = Xte(:,inds4(rind));
    c0(:,6) = Xte(:,inds5(rind));
    c0(:,7) = Xte(:,inds6(rind));
    c0(:,8) = Xte(:,inds7(rind));
    c0(:,9) = Xte(:,inds8(rind));
    c0(:,10) = Xte(:,inds9(rind));
    % These inititial centers are pretty ideal; yet the clustering still 
    % fails to recover the 10 digits. An alternative is to use:
    % rind = randi(m,[1,10]);
    % c0 = full(Xte(:,rind)); 
    % In our case, most values in centers will be non-zero, so full vectors
    % should perform better than sparse ones.
    [tmp_c,tmp_obj,tmp_y]=k_means(Xte,c0,T,1);
    if tmp_obj<obj
        c=tmp_c;
        obj=tmp_obj;
        y=tmp_y;
    end 
end
toc
for i = 1:10
    figure;
    imagesc(reshape(c(:,i),28,28)');
end