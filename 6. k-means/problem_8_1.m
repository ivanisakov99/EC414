%% Problem 8.1: k-means implementation
clear
close all

%% 8.1b
mu = [[2;2], [-2; 2], [0; -3.25]];
sigma = [0.02 0.05 0.07];
X = generate_dataset(50, mu, sigma);

tiledlayout(1,2);
% figure;
nexttile;
plot(X(:,1), X(:,2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Dataset','FontSize',20);


c0 = [[3, 3] ; [-4, -1]; [2, -4]];
T = 10;
k = 3;

[c, obj, y] = k_means_clustering(X, c0, T);

% figure;
nexttile;
for i = 1:k
    index = find(y == i);
    if(i == 1)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [1, 0, 0]);
    elseif(i == 2)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 1, 0]);
    elseif(i == 3)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
    end
    hold on;
end
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Clustered Dataset', 'FontSize',20);
legend('C1','C2','C3','Location','east');
print -dpng Q8_1b.png
%% 8.1c

c0 = [[-0.14, 2.61] ; [3.15, -0.84]; [-3.28, -1.58]];
T = 10;
k = 3;

[c, obj, y] = k_means_clustering(X, c0, T);

figure;
for i = 1:k
    index = find(y == i);
    if(i == 1)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [1, 0, 0]);
    elseif(i == 2)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 1, 0]);
    elseif(i == 3)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
    end
    hold on;
end
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Clustered Dataset','FontSize',20);
legend('C1','C2','C3','Location','east');
print -dpng Q8_1c.png
%% 8.1d
k = 3;
d = 2;
T = 10;


for i = 1:10
    c0 = 3*rand(k,d);
    [c, obji, yi] = k_means_clustering(X, c0, T);
    
    obj(i) = obji;
    y(:, i) = yi;
    
end

[minobj, idx] = min(obj,[], 2);
fprintf('The minimum objective function value is: %3.2f, at position %d\n', minobj, idx);

figure;
for i = 1:k
    index = find(y(:,idx) == i);
    if(i == 1)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [1, 0, 0]);
    elseif(i == 2)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 1, 0]);
    elseif(i == 3)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
    end
    hold on;
end
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Clustered Dataset','FontSize',20);
legend('C1','C2','C3','Location','east');
print -dpng Q8_1d.png
%% 8.1e
clear all
close all

T=70;

load mnist.mat

[d,m]=size(Xte); % Using Xte because it is conveniently smaller than Xtr 
c = 0; 
obj=inf;
% rng(0);
rng('default');
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
% for i = 1:10
%     figure;
%     imagesc(reshape(c(:,i),28,28)');
% end

tiledlayout(5,2);
j=0;
for i = 1:10
    j = j + 1;
    nexttile
    imagesc(reshape(c(:,i),28,28)');
    title(['C', num2str(j)]);
end
% 
print -dpng Q8_1e.png
%% 8.1f
clear
close all

num_cluster = 3;
points_per_cluster = 500*ones(num_cluster, 1);
[X, label] = sample_circle(num_cluster, points_per_cluster);

tiledlayout(1,2);
nexttile;
plot(X(:,1), X(:,2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Dataset','FontSize',20);

k = 3;
d = 2;
T = 10;


for i = 1:10
    c0 = 3*rand(k,d);
    [c, obji, yi] = k_means_clustering(X, c0, T);
    
    obj(i) = obji;
    y(:, i) = yi;
end
[minobj, idx] = min(obj,[], 2);

nexttile;
for i = 1:k
    index = find(y(:,idx) == i);
    if(i == 1)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [1, 0, 0]);
    elseif(i == 2)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 1, 0]);
    elseif(i == 3)
        plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
    end
    hold on;
end
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Clustered Dataset','FontSize',20);
legend('C1','C2','C3','Location','northeast');
print -dpng Q8_1f.png