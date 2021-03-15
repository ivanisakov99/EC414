%% 8.1b
clear
close all

mu = [[2;2], [-2; 2], [0; -3.25]];
sigma = [0.02 0.05 0.07];
X = generate_dataset(50, mu, sigma);


figure;
plot(X(:,1), X(:,2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Dataset','FontSize',20);

print -dpng Q8_1b1.png


c0 = [[3, 3] ; [-4, -1]; [2, -4]];
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
title('Clustered Dataset', 'FontSize',20);
legend('C1','C2','C3','Location','east');
print -dpng Q8_1b2.png
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
clear
close all
load('mnist.mat');

k = 10;
d = 784;
T = 10;
C = zeros(100, 784);
X = [Xtr;Xte];

for i = 1:10
    c0 = 3*rand(k,d);
    [c, obji, yi] = k_means_clustering(X, c0, T);
    C(((i-1)*10 + 1) : (i*10), :) = c;
    obj(i) = obji;
    y(:, i) = yi;
    fprintf('%d\n', i);
end
[minobj, idx] = min(obj,[], 2);

tiledlayout(5,2);
j=0;
for i = ((idx - 1)*10 + 1) : (idx*10)
    j = j + 1;
    nexttile
    imagesc(reshape(C(i,:),28,28)');
    title(['C', num2str(j)]);
end

print -dpng Q8_1e.png
%% 8.1f
clear
close all

num_cluster = 3;
points_per_cluster = 500*ones(num_cluster, 1);
[X, label] = sample_circle(num_cluster, points_per_cluster);

figure;
plot(X(:,1), X(:,2), 'x', 'LineWidth', 2, 'Color', [0, 0, 1]);
xlabel('x1','FontSize',20);
ylabel('x2','FontSize',20);
title('Dataset','FontSize',20);
print -dpng Q8_1f1.png

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
legend('C1','C2','C3','Location','northeast');
print -dpng Q8_1f2.png