function [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, u, y)
    N = length(y)-1;
    %% Inovate based on previous iterations parameters
    xkm1km1 = mu0;
    Pkm1km1 = zeros(size(mu0, 1));
    %% Kalman filter
    xkm1ks = zeros(size(mu0, 1), N);
    xkks = zeros(size(mu0, 1), N+1);
    xkks(:, 1) = mu0;
    Pkm1ks = cell(1, N);
    Pkks = cell(1, N+1);
    Pkks{1} = Pkm1km1;
    for k = 1:N
        xkm1k = A*xkm1km1+B*u(k);  % u(k) is ukm1
        ek = y(k+1)-C*xkm1k-D*u(k+1);
        Pkm1k = A*Pkm1km1*A.'+Q;
        Sigmak = C*Pkm1k*C.'+R;
        Kk = Pkm1k*C.'/Sigmak;
        tmp = Kk*C;
        Pkk = (eye(size(tmp))-tmp)*Pkm1k;
        xkk = xkm1k + Kk*ek;
        xkm1ks(:, k) = xkm1k;
        xkks(:, k+1) = xkk;
        Pkm1ks{k} = Pkm1k;
        Pkks{k+1} = Pkk;
        xkm1km1 = xkk;
        Pkm1km1 = Pkk;
    end
    
    %% Kalman smoother
    xNk = xkks(:, end);
    xNks = zeros(size(xNk, 1), N+1);
    xNks(:, end) = xNk;
    PNk = Pkks{end};
    PNks = cell(1, N+1);
    PNks{end} = PNk;
    for k = N:-1:1
        Jkm1 = Pkks{k}*A.'/Pkm1ks{k};
        xNkm1 = xkks(:, k)+Jkm1*(xNk-xkm1ks(:, k));
        PNkm1 = Pkks{k}+Jkm1*(PNk-Pkm1ks{k});
        xNk = xNkm1;
        xNks(:, k) = xNk;
        PNk = PNkm1;
        PNks{k} = PNk;
    end
    
    %% Lag-one Covariance Smoother
    tmp = Kk*C;
    PNkkm1 = (eye(size(tmp))-tmp)*A*PNks{end-1};
    PNkkm1s = cell(1, N);
    PNkkm1s{end} = PNkkm1;
    for k = N:-1:2
        Jkm1 = Pkks{k}*A.'/Pkm1ks{k};
        Jkm2t = (Pkks{k-1}*A.'/Pkm1ks{k-1}).';
        PNkm1km2 = Pkks{k}*Jkm2t+Jkm1*(PNkkm1-A*Pkks{k})*Jkm2t;
        PNkkm1 = PNkm1km2;
        PNkkm1s{k-1} = PNkkm1;
    end
end

