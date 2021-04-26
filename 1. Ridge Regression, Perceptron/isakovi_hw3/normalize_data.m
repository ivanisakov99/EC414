function [XtrainNormalized, XtestNormalized] = normalize_data(Xtrain, Xtest)
[m,n] =size(Xtrain);
    for i=1:n
        Xmean = mean(Xtrain);
        Xstd = std(Xtrain);
        XtrainNormalized = (Xtrain - Xmean)./Xstd;
        XtestNormalized = (Xtest - Xmean)./Xstd;
    end
end