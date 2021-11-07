function [A, B, C, D, mu0, Qtot, cost] = findModel(u, y, Su, Sy, nx, A, B, C, D, verbose, max_iters)
%FINDMODEL Takes in the measured input, output and their covariance matrix
%   Returns the ABCD parameters for the requested order, and also the total
%   state covariance matrix Qtot
    if exist('max_iters', 'var') ~= 1
        max_iters = 100;
    end
    %% Initial parameters
    Qtot = B*Su*B.';
    mu0 = zeros(nx, 1);
    %TODO
    params = toVect(A, B, C, D, mu0, Qtot);
    lastCost = costForParams(params, u, y, Sy, nx);
    if verbose
        fprintf("Log likelihood of initial guess: %.3f\n", lastCost);
    end
    for j = 1:max_iters
        [params] = optimizeIteration(u, y, params, Sy, nx);
        cost = costForParams(params, u, y, Sy, nx);
        relChange = abs((cost-lastCost)/lastCost);
        if verbose
            fprintf("(%u) New log likelihood: %.3f (relative change of %.2e percent)\n", j, cost, relChange*100);
        end
        if relChange < 1e-4 || isinf(cost)
            break
        end
        lastCost = cost;
    end
    [A, B, C, D, mu0, Qtot] = fromVect(params, nx, size(u, 1), size(y, 1));
end

function [ret] = toVect(A, B, C, D, mu0, Qtot)
    n = size(A, 1)+numel(B)+numel(C)+numel(mu0)+numel(Qtot);%numel(A)+numel(D);
    ret = zeros(1, n);
    n1 = size(A, 1);%numel(A);
    idx = 1;
    ret(1:n1) = diag(A);%reshape(A, 1, []);
    idx = idx + n1;
    n1 = numel(B);
    ret(idx-1+(1:n1)) = reshape(B, 1, []);
    idx = idx + n1;
    n1 = numel(C);
    ret(idx-1+(1:n1)) = reshape(C, 1, []);
    idx = idx + n1;
%     n1 = numel(D);
%     ret(idx+(1:n1)) = reshape(D, 1, []);
%     idx = idx + n1;
    n1 = numel(mu0);
    ret(idx-1+(1:n1)) = reshape(mu0, 1, []);
    idx = idx + n1;
    n1 = numel(Qtot);
    ret(idx:end) = reshape(Qtot, 1, []);
end

function [A, B, C, D, mu0, Qtot] = fromVect(params, nx, nu, ny)
    idx = 1;
    n = nx;%*nx;
    A = diag(params(idx-1+(1:n)));%reshape(params(idx-1+(1:n)), nx, nx);
    idx = idx + n;
    n = nx*nu;
    B = reshape(params(idx-1+(1:n)), nx, nu);
    idx = idx + n;
    n = ny*nx;
    C = reshape(params(idx-1+(1:n)), ny, nx);
    idx = idx + n;
%     n = nx*nu;
%     D = reshape(params(idx-1+(1:n)), nx, nu);
%     idx = idx + n;
    D = zeros(ny, nu);
    n = nx;
    mu0 = reshape(params(idx-1+(1:n)), nx, 1);
    idx = idx + n;
    n = nx*nx;
    Qtot = reshape(params(idx:end), nx, nx);
end

function [paramsOut] = optimizeIteration(u, y, params, R, nx)
    h = 1e-6;
    J = zeros(length(params));
    dx = zeros(length(params), 1);
    for i = 1:length(params)
        for j = i:length(params)
            if i == j
                ptmp = params;
                v = costForParams(params, u, y, R, nx);
                ptmp(i) = ptmp(i)-h;
                vm1 = costForParams(ptmp, u, y, R, nx);
                ptmp(i) = params(i)+h;
                vp1 = costForParams(ptmp, u, y, R, nx);
                J(i, i) = (vp1+vm1-2*v)/(h*h);
                dx(i) = (vp1-vm1)/(2*h);
                continue
            end
            %g = dC/di => dg/dj=(gp1-gm1)/(2h)
            %gp1 = g(j+1) etc.
            ptmp = params;
            ptmp(j) = ptmp(j) + h;
            ptmp(i) = ptmp(i) + h;
            vgp1p1 = costForParams(ptmp, u, y, R, nx);
            ptmp(i) = params(i) - h;
            vgp1m1 = costForParams(ptmp, u, y, R, nx);
            gp1 = (vgp1p1-vgp1m1)/(2*h);
            ptmp(j) = params(j) - h;
            vgm1m1 = costForParams(ptmp, u, y, R, nx);
            ptmp(i) = params(i) + h;
            vgm1p1 = costForParams(ptmp, u, y, R, nx);
            gm1 = (vgm1p1-vgm1m1)/(2*h);
            J(i, j) = (gp1-gm1)/(2*h);
            J(j, i) = J(i, j);
        end
    end
    lambda = 0;
    if any(isnan(J), 'all') || any(isinf(J), 'all')
        paramsOut = params;
%         for i = 1:length(params)
%             for j = i:length(params)
%                 if i == j
%                     ptmp = params;
%                     v = costForParams(params, u, y, R, nx);
%                     ptmp(i) = ptmp(i)-h;
%                     vm1 = costForParams(ptmp, u, y, R, nx);
%                     ptmp(i) = params(i)+h;
%                     vp1 = costForParams(ptmp, u, y, R, nx);
%                     J(i, i) = (vp1+vm1-2*v)/(h*h);
%                     dx(i) = (vp1-vm1)/(2*h);
%                     continue
%                 end
%                 %g = dC/di => dg/dj=(gp1-gm1)/(2h)
%                 %gp1 = g(j+1) etc.
%                 ptmp = params;
%                 ptmp(j) = ptmp(j) + h;
%                 ptmp(i) = ptmp(i) + h;
%                 vgp1p1 = costForParams(ptmp, u, y, R, nx);
%                 ptmp(i) = params(i) - h;
%                 vgp1m1 = costForParams(ptmp, u, y, R, nx);
%                 gp1 = (vgp1p1-vgp1m1)/(2*h);
%                 ptmp(j) = params(j) - h;
%                 vgm1m1 = costForParams(ptmp, u, y, R, nx);
%                 ptmp(i) = params(i) + h;
%                 vgm1p1 = costForParams(ptmp, u, y, R, nx);
%                 gm1 = (vgm1p1-vgm1m1)/(2*h);
%                 J(i, j) = (gp1-gm1)/(2*h);
%                 J(j, i) = J(i, j);
%             end
%         end
        return;
    end
    [U, S, V] = svd(J);
    tmp = diag(S);
    tmp2 = tmp.*tmp + lambda*lambda;
    tmp3 = tmp./tmp2;
    tmp3(isnan(tmp3)) = 0;
    Lambda = diag(tmp3);
    currentCost = costForParams(params, u, y, R, nx);%sum(abs(dx));
    delta = -V*Lambda*U.'*dx;
    paramsOut = params + delta.';
%     for i = 1:length(params)
%         ptmp = paramsOut;
%         ptmp(i) = ptmp(i)-h;
%         vm1 = costForParams(ptmp, u, y, R, nx);
%         ptmp(i) = paramsOut(i)+h;
%         vp1 = costForParams(ptmp, u, y, R, nx);
%         dx2(i) = (vp1-vm1)/(2*h);
%     end
    v = costForParams(paramsOut, u, y, R, nx);%sum(abs(dx2));
    if v < currentCost || true
        lambda = max(tmp)/100;
        amt = 0;
        while v < currentCost
            tmp2 = tmp.*tmp + lambda*lambda;
            Lambda = diag(tmp./tmp2);
            delta = -V*Lambda*U.'*dx;
            paramsOut = params + delta.';
            lambda = lambda * 10;
%             for i = 1:length(params)
%                 ptmp = paramsOut;
%                 v = costForParams(paramsOut, u, y, R, nx);
%                 ptmp(i) = ptmp(i)-h;
%                 vm1 = costForParams(ptmp, u, y, R, nx);
%                 ptmp(i) = paramsOut(i)+h;
%                 vp1 = costForParams(ptmp, u, y, R, nx);
%                 J(i, i) = (vp1+vm1-2*v)/(h*h);
%                 dx2(i) = (vp1-vm1)/(2*h);
%             end
            v = costForParams(paramsOut, u, y, R, nx);%sum(abs(dx2));
            if amt == 5
                paramsOut = params;
            end
            amt = amt+1;
        end
    end
end

function [ret] = costForParams(params, u, y, R, nx)
    [A, B, C, D, mu0, Qtot] = fromVect(params, nx, size(u, 1), size(y, 1));
    for i = 1:nx
        Qtot(i, i) = abs(Qtot(i, i));
    end
    %[sigmaKs, eKs] = kalman2(u, y, A, B, C, D, mu0, Qtot, R);
    ret = cost2(A, B, C, D, mu0, u, y)+kalmanCost(u, y, A, B, C, D, mu0, Qtot, R);%cost(sigmaKs, eKs);
end

function [ret] = cost(sigmaKs, eKs)
    ret = 0;
    for k = 1:length(sigmaKs)
        ek = eKs(:, k);
        ret = ret - (log(abs(det(sigmaKs{k})))+ek.'/sigmaKs{k}*ek);
    end
end

function [ret] = cost2(A, B, C, D, mu0, u, y)
    %t = 0:(size(u, 2)-1);
    y2 = zeros(size(y));
    x = zeros(size(A, 1), size(y, 2));
    x(:, 1) = mu0;
    for k = 1:size(y, 2)
        y2(:, k) = C*x(:, k);
        if sum(abs(y2(:, k))) > 1e10
            ret = -inf;
            return;
        end
        x(:, k+1) = A*x(:, k)+B*u(k);
    end
    ret = -sum((y2-y).^2, 'all');
end

function [ret] = kalmanCost(u, y, A, B, C, D, mu0, Q, R)
    N = size(y, 2)-1;
    %% Inovate based on previous iterations parameters
    xkm1km1 = mu0;
    Pkm1km1 = zeros(size(mu0, 1));
    %% Kalman filter
    I = eye(size(A));
    ret = 0;
    for k = 1:N
        xkm1k = A*xkm1km1+B*u(k);  % u(k) is ukm1
        ek = y(k+1)-C*xkm1k-D*u(k+1);
        Pkm1k = A*Pkm1km1*A.'+Q;
        Sigmak = C*Pkm1k*C.'+R;
        Kk = Pkm1k*C.'/Sigmak;
        tmp = Kk*C;
        Pkk = (I-tmp)*Pkm1k;
        xkk = xkm1k + Kk*ek;
        xkm1km1 = xkk;
        Pkm1km1 = Pkk;
        ret = ret - (log(abs(det(Sigmak)))+ek.'/Sigmak*ek);
    end
end
