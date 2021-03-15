function [X_poly] = generate_poly_features(X,p)

X_poly = X;

if(p > 1)
    for i=2:p
        X_poly = [X_poly, X.^i];
    
    end
end

end

