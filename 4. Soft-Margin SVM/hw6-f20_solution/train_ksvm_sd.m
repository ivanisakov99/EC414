function [alpha, b, obj, zeroOneAverageLoss] = train_ksvm_sd(X,y,Tmax,c, gamma)

[m,d]=size(X);

xsq = sum(X.*X,2);
K = xsq-((2*X)*X');
K = K + xsq';   % vector version of (a-b)^2 = a^2+b^2-2ab. This makes K(i,j) = norm(xi-xj)^2
K = exp(-gamma*K);
K_zeros=[ zeros(1,m+1); zeros(m,1) K];
Kext = [ones(1,m);K];

obj=zeros(Tmax,1);
zeroOneAverageLoss=zeros(Tmax,1);
tildeAlpha=zeros(m+1,1);
yt = y';

for i=1:Tmax
    yAlphaKext = yt.*(tildeAlpha'*Kext);         % same dimensions as y', stores the values of y_j*alpha*Kext_j
    zeroOneAverageLoss(i)= sum(yAlphaKext<0)/m;  % each negative value is a misclassification
    tmp = 1 - yAlphaKext;                        % hinge losses, but without filtering negative values
    K_zerosAlpha = K_zeros*tildeAlpha;
    obj(i)=0.5*(tildeAlpha'*K_zerosAlpha)+c*sum(tmp(tmp>0)); % tmp>0 filters out negative values
    g=K_zerosAlpha-c*(Kext*(y.*(tmp>0)')); % tmp>0 also serves as a filter here
    eta_t=1/i;
    tildeAlpha=tildeAlpha - eta_t * g;
end

b=tildeAlpha(1);
alpha=tildeAlpha(2:end);

end
