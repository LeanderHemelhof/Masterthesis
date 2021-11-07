function [A, B, C, D, Q, R, S] = SSEstim(y, u, n)
    assert(size(y, 2) == size(u, 2))
    N = size(y, 2);
    m = size(u, 1);
    l = size(y, 1);
    %choose i and j so 2i+j<=N and i>n
    %check if 2i+j-2 is enough (y0 and u0 vs y1 and u1)
    i = floor(N/(2*(m+l+1)));%floor(N/3);
    j = N-(2*i);
    U02im1 = BH(u, 2*i, j);
    Y02im1 = BH(y, 2*i, j);
    H = [U02im1;Y02im1]/sqrt(j);
    R1 = specialRQ(H);
%     [Q1, R1] = qr(H.');
%     R1 = R1.';
    R5614 = R1(2*m*i+l*i+(1:l*i), 1:2*m*i+l*i);
    R1414 = R1(1:2*m*i+l*i, 1:2*m*i+l*i);
    L = R5614/R1414;
    Li1 = L(:, 1:m*i);
    Li2 = L(:, m*i+(1:m*i));
    Li3 = L(:, 2*m*i+(1:l*i));
    
    [U, S, ~] = svd(Li1+Li3*Li2);%svd([Li1 zeros(size(L, 1), m*i) Li3]*R1414);
    U1 = U(:, 1:n);
    U1bar = U1(1:end-l, :);
    S1 = S(1:n, 1:n);
    rS1 = S1^(-0.5);
    R6614 = R1(2*m*i+l*(i+1)+(1:l*(i-1)), 1:2*m*i+l*i);
    R5514 = R1(2*m*i+l*i+(1:l), 1:2*m*i+l*i);
    M1 = [rS1*pinv(U1bar)*R6614;R5514];
    R2314 = R1(m*i+(1:m*i), 1:2*m*i+l*i);
    M2 = [rS1*pinv(U1)*R5614;R2314];
    K = M1*pinv(M2);
    A = K(1:n, 1:n);
    C = K(n+1:end, 1:n);
    Gammai = U1*S1^0.5;
    Gammaim1 = Gammai(1:end-l, :);
    AG = A*pinv(Gammai);
    CG = C*pinv(Gammai);
    subK = K(:, n+1:end);
    K1211 = subK(:, 1:m);
    %Trans = [eye(n)-AG(:, l+1:end)*Gammaim1 -AG(:, 1:l);-CG(:, l+1:end)*Gammaim1 eye(l)-CG(:, 1:l)];
    Trans = [eye(n)-AG(:, l+1:end)*Gammaim1;-CG(:, l+1:end)*Gammaim1];
    %BD = Trans\K1211;
    B = Trans\K1211;%BD(1:n, :);
    D = zeros(l, m);
    %D = BD(n+1:end, :);
%     Zi = L*H(1:end-(l*i), :);
%     
%     R6615 = R1(2*m*i+l*(i+1)+(1:l*(i-1)), 1:2*m*i+l*(i+1));
%     R1515 = R1(1:2*m*i+l*(i+1), 1:2*m*i+l*(i+1));
%     Zip1 = R6615/R1515*H(1:end-(l*(i-1)), :);
    rho = M1-K*M2;
    QRS = rho*rho.'/j;
    Q = QRS(1:n, 1:n);
    S = QRS(1:n, n+1:end);
    R = QRS(n+1:end, 1+n:end);
end

function [ret] = BH(xs, rows, cols)
    h = size(xs, 1);
    ret = zeros(h*rows, cols);
    y = 0;
    for r=1:rows
        ret(y+(1:h), :) = xs(:, r-1+(1:cols));
        y = y+h;
    end
end

function [R] = specialRQ(H)
    assert(size(H, 2)>=size(H, 1))
    Qt = zeros(size(H));
    R = zeros(size(H, 1));
    for row = 1:size(H, 1)
        ai = H(row, :);
        ui = ai;
        for i=1:row-1
            R(row, i) = Qt(i, :)*ai.';
            ui = ui - R(row, i)*Qt(i, :);
        end
        ei = ui/sqrt(ui*ui.');
        Qt(row, :) = ei;
        R(row, row) = ei*ai.';
    end
end