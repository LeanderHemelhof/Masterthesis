function [sigmaKs, eKs] = kalman2(u, y, A, B, C, D, mu0, Q, R)
    N = size(y, 2)-1;
    %% Inovate based on previous iterations parameters
    xkm1km1 = mu0;
    Pkm1km1 = zeros(size(mu0, 1));
    %% Kalman filter
%     xkm1ks = zeros(size(mu0, 1), N);
%     xkks = zeros(size(mu0, 1), N+1);
%     xkks(:, 1) = mu0;
%     Pkm1ks = cell(1, N);
%     Pkks = cell(1, N+1);
%     Pkks{1} = Pkm1km1;
    sigmaKs = cell(1, N);
    eKs = zeros(length(mu0), N);
    for k = 1:N
        xkm1k = A*xkm1km1+B*u(k);  % u(k) is ukm1
        ek = y(k+1)-C*xkm1k-D*u(k+1);
        Pkm1k = A*Pkm1km1*A.'+Q;
        Sigmak = C*Pkm1k*C.'+R;
        Kk = Pkm1k*C.'/Sigmak;
        tmp = Kk*C;
        Pkk = (eye(size(tmp))-tmp)*Pkm1k;
        xkk = xkm1k + Kk*ek;
%         xkm1ks(:, k) = xkm1k;
%         xkks(:, k+1) = xkk;
%         Pkm1ks{k} = Pkm1k;
%         Pkks{k+1} = Pkk;
        xkm1km1 = xkk;
        Pkm1km1 = Pkk;
        sigmaKs{k} = Sigmak;
        eKs(:, k) = ek;
    end
end

