function [A, B, C, D, Q, R, S] = SSEstim2(y, u, nmax)
    % source: http://www.diva-portal.org/smash/get/diva2:316017/FULLTEXT02.pdf
    % nreal<=nmax
    n = nmax;
    Ntot = size(y, 2);
    % l>=r>nreal
    % any s possible, but estimates of Q, R and S only correct for infinity
    r = nmax+1;
    l = r;
    s = l;
    N = Ntot-l-s-1;  % Need l ahead and s+1 behind
    ts = (s+2):(Ntot-l);
    %% Yhatslp1
    phis = phi(u, y, s, ts, 0);
    phitot = [phis;phiTilde(u, l+1, ts, 0)];
    p = size(y, 1);
    m = size(u, 1);
    Yr = getYr(y, r, ts, 0);
    M = Yr/phitot;
    Theta = M(:, 1:s*(p+m));
    assert(max(abs(Yr-M*phitot), [], 'all') < 1)
    Yhatslp1t = Theta*phis;
    Ybarslp1t = M*phitot;
    
    %% Ybarsp1ltp1
    phis = phi(u, y, s+1, ts, 1);
    phitot = [phis;phiTilde(u, l, ts, 1)];
    Yrtp1 = getYr(y, r, ts, 1);
    M = Yrtp1/phitot;
    Ybarsp1ltp1 = M*phitot;
    [U, S, V] = svd(Yhatslp1t);
    U1 = U(:, 1:n); S1 = S(1:n, 1:n);
    L = pinv(U1*sqrtm(S1));
    xbartp1 = L*Ybarsp1ltp1;
    xt = L*Ybarslp1t;
    philp1 = phiTilde(u, l+1, ts, 0);
    vals = [xt; philp1];
    M = [xbartp1;y(ts)]/vals;
    A = M(1:n, 1:n);
    C = M(n+1:end, 1:n);
    beta = M(:, n+1:end);
    coefs = beta(n+1:end, :);
    Dd = coefs(:, 1:m);
    tmp = [C];
    last = C;
    coefs2 = [coefs(:, m+(1:m))];
    id = 2*m;
    for k=2:l
        tmp = [tmp;last*A];
        last = last*A;
        coefs2 = [coefs2;coefs(:, id+(1:m))];
        id = id+m;
    end
    
    B = tmp\coefs2;
    M = y(:, ts)/u(:, ts);
    %B = (pinv(C)*y(:, ts-1)-A*pinv(C)*y(:, ts))/u(:, ts);
    D = zeros(p, m);
    E = [xbartp1;y(ts)]-[A B;C D]*[xt;u(:, ts)];
end

function [pt] = phiTilde(u, l, ts, toffset)
    m = size(u, 1);
    pt = zeros(m*l, length(ts));
    col = 1;
    for t=ts
        rowOffset = 0;
        for i=1:l
            pt(rowOffset+(1:m), col) = u(:, t+toffset+i-1);
            rowOffset = rowOffset+m;
        end
        col = col+1;
    end
end

function [ps] = phi(u, y, s, ts, toffset)
    p = size(y, 1);
    m = size(u, 1);
    ps = zeros(s*(p+m), length(ts));
    col = 1;
    for t=ts
        rowOffset = 0;
        for i=1:s
            ps(rowOffset+(1:p), col) = y(:, t+toffset-i);
            ps(rowOffset+p+(1:m), col) = u(:, t+toffset-i);
            rowOffset = rowOffset+m+p;
        end
        col = col+1;
    end
end

function [Yr] = getYr(y, r, ts, toffset)
    p = size(y, 1);
    Yr = zeros(r*p, length(ts));
    col = 1;
    for t=ts
        rowOffset = 0;
        for k=0:(r-1)
            Yr(rowOffset+(1:p), col) = y(:, t+toffset+k);
            rowOffset = rowOffset+p;
        end
        col = col+1;
    end
end