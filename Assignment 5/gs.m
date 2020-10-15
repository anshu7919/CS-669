function out = gs(p,q,var)
%     d = size(p);
    out = 1/((2*pi)*det(var)^2)*exp(-(p-q)*inv(var)*(p-q)'/(2));
    return

    