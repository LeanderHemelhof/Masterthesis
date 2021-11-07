function [A, B, C, D, Q, R, S, K, Rr] = SSEstim3(y, u, nmax)
    [ny, N] = size(y);
    [nu, N2] = size(u);
    assert(N==N2);
    i = 20*nmax/nu;
    j = N-2*i+1;
    U = blockHankel(u/sqrt(j), 2*i, j);
    r = rank(U*U.');
    if r ~= 2*nu*i
        fprintf("Input signal excitation order too low for max model order. Max order possible is %u.\n", floor(r/4))
    end
    Y = blockHankel(y/sqrt(j), 2*i, j);
    UY = [U;Y];
    
    %[a3,b3,c3,d3,~, ~, ~] = subid(y,u,i,1,[],[],1, blockHankel(u/sqrt(j), 2*i, j));
    
    R = triu(qr(UY.')).';
    R = R(1:2*i*(ny+nu), 1:2*i*(ny+nu));
    R5614 = R(2*nu*i+ny*i+1:end, 1:2*nu*i+ny*i);
    R1414 = R(1:2*nu*i+ny*i, 1:2*nu*i+ny*i);
    L = R5614*pinv(R1414);
    LUp = L(:, 1:nu*i);
    LYp = L(:, 2*i*nu+1:end);
    R1113 = R(1:nu*i, 1:2*nu*i);
    R4413 = R(2*nu*i+(1:ny*i), 1:2*nu*i);
    R2313 = R(nu*i+1:2*nu*i, 1:2*nu*i);
    Pi = eye(2*nu*i)-R2313.'/(R2313*R2313.')*R2313;
    %test = (LUp*R(1:nu*i, 1:(2*nu+ny)*i)+LYp*R(2*nu*i+(1:ny*i), 1:(2*nu+ny)*i));
    [U, S, V] = svd([(LUp*R1113+LYp*R4413)*Pi, LYp*R(2*nu*i+(1:ny*i), 2*nu*i+(1:ny*i))]);
    [Utst, Stst, Vtst] = denoise([(LUp*R1113+LYp*R4413)*Pi, LYp*R(2*nu*i+(1:ny*i), 2*nu*i+(1:ny*i))]);
    [Utst2, Stst2, Vtst2] = svd(shrink([(LUp*R1113+LYp*R4413)*Pi, LYp*R(2*nu*i+(1:ny*i), 2*nu*i+(1:ny*i))], 3));
    n = nmax;
    U1 = U(:, 1:n);
    S1 = S(1:n, 1:n);
    Gammai = U1*sqrtm(S1);
    Gammaim1 = Gammai(1:end-ny, :);
    R6615 = R(2*i*nu+ny*(i+1)+(1:ny*(i-1)), 1:2*i*nu+ny*(i+1));
    R5515 = R(2*i*nu+ny*i+(1:ny), 1:2*i*nu+ny*(i+1));
    Tl = [pinv(Gammaim1)*R6615;R5515];
    R5615 = R(2*i*nu+ny*i+(1:ny*i), 1:2*i*nu+ny*(i+1));
    R2315 = R(nu*i+(1:nu*i), 1:2*i*nu+ny*(i+1));
    Tr = [pinv(Gammai)*R5615;R2315];
    S = Tl/Tr;
    AC = S(:, 1:n);
    A = AC(1:n, :);
    C = AC(n+1:end, :);
    residue = Tl-S*Tr;
    
    %% Recalculate Gamma and Gamma_bar using the found A and C instead of the estimation made before
    Gammai(1:ny, :) = C;
    prev = C;
    for k=2:i
        prev = prev*A;
        Gammai((k-1)*ny+(1:ny), :) = prev;
    end
    invGammai = pinv(Gammai);
    Gammaim1 = Gammai(1:end-ny, :);
    Tl = [pinv(Gammaim1)*R6615;R5515];
    Tr = [invGammai*R5615;R2315];
    
    P = Tl-AC*pinv(Gammai)*R5615;
    Q = R2315;
    
    L1 = A*invGammai;
    L2 = C*invGammai;
    M = pinv(Gammaim1);
    zpM = [zeros(n, ny) M];
    Nbase = [eye(ny), zeros(ny, n);zeros(ny*(i-1), ny) Gammaim1];
    tot = zeros([size(Q, 2), nu]*(n+ny));
    %% Calculate Kronecker product needed for B and D
    for k=1:i
        Nk = [zpM(:, ((k-1)*ny+1):end)-L1(:, ((k-1)*ny+1):end), zeros(n, (k-1)*ny);-L2(:, ((k-1)*ny+1):end), zeros(ny, (k-1)*ny)];
        if k==1
            Nk(n+1:end, 1:ny) = Nk(n+1:end, 1:ny)+eye(ny);
        end
        Nk = Nk*Nbase;
        tot = tot + KroneckerProduct(Q((k-1)*nu+1:k*nu, :).', Nk);
    end
    vecDB = tot\P(:);
    DB = reshape(vecDB, n+ny, nu);
    D = DB(1:ny, :);
    B = DB(ny+1:end, :);
    
    %% Calculate Q, R and S
    QRS = residue*residue.';
    Q = QRS(1:n, 1:n);
    R = QRS(n+1:end, n+1:end);
    S = QRS(1:n, n+1:end);
    
    %% Calculate K and R
    [K, Rr] = findKandR(A, C, Q, R, S);
end

function [M] = denoiseBH(H, Pi, bw, bh, r)
    H1 = H;
    maxCnt = 20;
    cnt = 0;
    while cnt < maxCnt
        H2 = shrink(H1*Pi, r)+H1*(eye(size(Pi))-Pi);
        H1 = projToBH(H2, bw, bh);
        if norm(H1-H2, 'fro') < 5e-3*norm(H1, 'fro')
            break
        end
        cnt = cnt + 1;
    end
    if cnt == maxCnt
        fprintf('denoiseBH: convergence took too long. End result: %.3f > %.3f\n', norm(H1-H2, 'fro'), 5e-3*norm(H1, 'fro'))
    end
    M = H1;
end

function [M] = shrink(Y, r)
    [U, S, V] = svd(Y);
    [n, m] = size(Y);
    svs = diag(S);
    S2 = diag(svs(r+1:end));
    k = size(S2, 1);
    Snew = zeros(size(S));
    for i=1:r
        si = svs(i);
        v1 = inv(si*si*eye(k)-S2*S2');
        tmp1 = sum(diag(si*v1))/n;
        v2 = inv(si*si*eye(k)-S2'*S2);
        tmp2 = sum(diag(si*v2))/m;
        D = tmp1*tmp2;
        Dp = tmp1/m*sum(diag(-2*si*si*v2*v2+v2))+tmp2/n*sum(diag(-2*si*si*v1*v1+v1));
        Snew(i, i) = -2*D/Dp;
    end
    M = U*Snew*V';
end

function [M] = projToBH(H, bw, bh)
    [n, m] = size(H);
    M = zeros(size(H));
    brs = n/bh; bcs = m/bw;
    %#skews = brs+(bcs-1)=K1+K2
    for sid = 1:brs
        tot = zeros(bh, bw);
        cnt = 0;
        for rid=(sid-1)*bh+1:-bh:1
            if (cnt+1)*bw > m
                break
            end
            tot = tot+H(rid+(0:bh-1), cnt*bw+(1:bw));
            cnt = cnt+1;
        end
        tot = tot/cnt;
        cnt = 0;
        for rid=(sid-1)*bh+1:-bh:1
            if (cnt+1)*bw > m
                break
            end
            M(rid+(0:bh-1), cnt*bw+(1:bw)) = tot;
            cnt = cnt+1;
        end
    end
    for sid2 = 1:(bcs-1)
        tot = zeros(bh, bw);
        cnt = 0;
        for cid=sid2*bw+1:bw:m
            tot = tot+H(n-(cnt+1)*bh+(1:bh), cid+(0:bw-1));
            cnt = cnt+1;
            if cnt==brs
                break
            end
        end
        tot = tot/cnt;
        cnt = 0;
        for cid=sid2*bw+1:bw:m
            M(n-(cnt+1)*bh+(1:bh), cid+(0:bw-1)) = tot;
            cnt = cnt+1;
            if cnt==brs
                break
            end
        end
    end
end

function [U, S, V] = denoise(Y, sig)
    %https://sites.lsa.umich.edu/hderksen/wp-content/uploads/sites/614/2018/05/A.I.a.47.pdf
    % Y=X+sigma*noise
    [n, m] = size(Y);
    if ~exist('sig', 'var')
        sig = 1/sqrt(m);
    end
    [U, Sy, V] = svd(Y/(sqrt(m)*sig));
    b = n/m;
    bound = 1+sqrt(b);
    Sy(Sy<bound) = 0;
    d = diag(Sy);
    S = zeros(size(Sy));
    for idx=1:sum(d~=0)
        di2 = d(idx)*d(idx);
        lxi2 = 0.5*((di2-1-b)+sqrt((di2-1-b)^2-4*b));
        ti2 = (1-b/(lxi2*lxi2))/(1+b/lxi2);
        pi2 = (1-b/(lxi2*lxi2))/(1+1/lxi2);
        S(idx, idx) = sqrt(lxi2*ti2*pi2)*sqrt(m)*sig;
    end
end

function [H] = blockHankel(lst, rows, cols)
    amt = size(lst, 1);
    H = zeros(amt*rows, cols);
    for row=1:rows
        H(amt*(row-1)+(1:amt), :) = lst(:, row:row+cols-1);
    end
end

