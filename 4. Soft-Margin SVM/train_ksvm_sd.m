function [alpha, b, obj, zeroOneAverageLoss] = train_ksvm_sd(X, y, T, c, gamma)
[m, ~] = size(X);
alpha_tilde = zeros(m + 1, 1);


obj = zeros(T, 1);
eta = 1/T;
zeroOneAverageLoss = zeros(T, 1);

K = compute_kernel(X, gamma);
K_tilde = [zeros(m + 1, 1), [zeros(1, m) ; K]];


for i=1:T
    obj(i) = obj_function(K, y, alpha_tilde, c);
    for j=1:m
        if (y(j)*alpha_tilde'*K_tilde(:,j)<=0)
            
            zeroOneAverageLoss(i) = zeroOneAverageLoss(i) + 1;
        end
    end
    
    subg_1 = K_tilde * alpha_tilde;
    subg_2 = subg_hinge(K, y, alpha_tilde, c);
    subg = subg_1 + subg_2;
    alpha_tilde = alpha_tilde - eta * subg;
    
end

b = alpha_tilde(1, :);
alpha = alpha_tilde(2 : end, :);
zeroOneAverageLoss = zeroOneAverageLoss/m;

end

function val = obj_function(K, y, alpha_tilde, c)
d = length(K);
K_tilde = [ones(1, d) ; K];
K2 = [zeros(d + 1, 1), [zeros(1, d); K]];
val = 0.5 * (alpha_tilde' * K2 * alpha_tilde) + c * sum(hinge(y' .* (alpha_tilde' * K_tilde)));
end

function val = hinge(t)
zeros(length(t), 1);
idx = find(t < 1);
val(idx) = 1 - t(idx);
end

function val = subg_hinge(K, y, alpha_tilde, c)
d = length(K);
K_tilde = [ones(1, d); K];
val = zeros(d, 1);
temp = zeros(d + 1, d);
cond = y' .*(alpha_tilde' * K_tilde);
idx = find(cond < 1);
temp(:, idx) = -c * y(idx)' .* K_tilde(:, idx);
val = sum(temp, 2);
end

function [K] = compute_kernel(X,gamma)
normX = full(sum(X.^2,2));
K = repmat(normX,1,size(X,1)) + repmat(normX',size(X,1),1) - 2*full(X*X');
K=exp(-K*gamma);
end
