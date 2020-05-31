function quad = simpsonQuad(L,T)
    q = size(L,2);  
    dim = size(L,1);
    Tq = T(1,q);
    T1 = T(1,1);
    mask = ones(size(q));
    mask(:,2:2:q) = 4 * ones(dim,1);
    mask(:,3:2:q) = 2 * ones(dim,1);
    mask(:,q) = ones(dim,1);
    quad = (Tq- T1)* sum(L.*mask)/(3*(q-1));

end
