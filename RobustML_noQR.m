function [A, B, C, D, Q, R, S, mu, Lambda] = RobustML_noQR(A, B, C, D, Q, R, S, mu, Lambda, y, u)
    
    [ny, N] = size(y);
    n = size(A, 1);
    nu = size(B, 2);
    tries = 0;
    it = 0;
    while tries ~= 100%< 1000
        %% Initialize Kalman recursion
        rootPtts = zeros([size(A) N]);
        %rootPtts(:, :, 1) = sqrtm(P1);
        xtts = zeros(size(mu, 1), N);
        xttm1s = zeros(size(mu, 1), N);
%         xtts(:, 1) = mu;
%         xttm1s(:, 1) = mu;
        
        %% Robust forward
        SR = S/R;
        Abar = A - SR*C;
        Bbar = B - SR*D;
        Qbar = Q - SR*S.';
        
        rootQ = sqrtm(Qbar);
        if max(abs(imag(rootQ)), [], 'all') ~= 0
            rootQ = real(rootQ);
        end
        rootR = sqrtm(R);
        rootPtp1ts = zeros([size(A) N-1]);
        
        rootP00 = sqrtm(Lambda);
        [~, R1] = qr([(Abar*rootP00).';rootQ.']);
        rootPttm1 = R1(1:n, :).';
        Pttm1 = rootPttm1*rootPttm1.';
        P10 = Pttm1;
        %P102 = A*Lambda*A.'+Q;
        xttm1 = (Abar+SR*C)*mu;
        [~, R2] = qr([rootR, C*rootPttm1;zeros(n, ny), rootPttm1].');
        rootPtts(:, :, 1) = R2(ny+1:end, ny+1:end).';
        Kt = Pttm1*C.'/(C*Pttm1*C.'+R);

        xtts(:, 1) = xttm1+Kt*(y(:, 1)-C*xttm1-D*u(:, 1));
        xttm1s(:, 1) = xttm1;
        for t=2:N
            [~, R1] = qr([(Abar*rootPtts(:, :, t-1)).';rootQ.']);
            rootPttm1 = R1(1:n, :).';
            Pttm1 = rootPttm1*rootPttm1.';
            %test1 = A*(rootPtts(:, :, t-1)*rootPtts(:, :, t-1).')*A.'+Q;
            rootPtp1ts(:, :, t-1) = rootPttm1;
            xttm1 = Abar*xtts(:, t-1)+Bbar*u(:, t-1)+SR*y(:, t-1);
            [~, R2] = qr([rootR C*rootPttm1;zeros(n, ny) rootPttm1].');
            rootPtts(:, :, t) = R2(ny+1:end, ny+1:end).';
            Ptt = rootPtts(:, :, t)*rootPtts(:, :, t).';
            Kt = Pttm1*C.'/(C*Pttm1*C.'+R);
            %test2 = Pttm1-Kt*C*Pttm1;
            
            xtts(:, t) = xttm1+Kt*(y(:, t)-C*xttm1-D*u(:, t));
            xttm1s(:, t) = xttm1;
        end
        
        %% Robust backward
        rootPtNs = zeros(size(rootPtts));
        PtNs = zeros([size(A) N]);
        xtNs = zeros(size(mu, 1), N);
        MtNs = zeros([size(A) N]);
        
        rootPtNs(:, :, N) = rootPtts(:, :, N);
        PtNs(:, :, N) = rootPtts(:, :, N)*rootPtts(:, :, N).';
        xtNs(:, N) = xtts(:, N);
        MtNs(:, :, N) = (eye(n)-Kt*C)*Abar*(rootPtts(:, :, N-1)*rootPtts(:, :, N-1).');
        for t=N-1:-1:1
            rootPtt = rootPtts(:, :, t);
            Ptt = rootPtt*rootPtt.';
            rootPtp1t = rootPtp1ts(:, :, t);
            rootPtp1N = rootPtNs(:, :, t+1);
            Jt = Ptt*Abar.'/(rootPtp1t*rootPtp1t.');
            [~, R3] = qr([rootPtt.'*Abar.' rootPtt.'; rootQ.' zeros(n); zeros(n) rootPtp1N.'*Jt.']);
            rootPtNs(:, :, t) = R3(n+(1:n), n+1:end).';
            
            PtNs(:, :, t) = rootPtNs(:, :, t)*rootPtNs(:, :, t).';
%             Ptp1t = rootPtp1t*rootPtp1t.';
%             test1 = rootPtts(:, :, t)*rootPtts(:, :, t).'+Jt*(PtNs(:, :, t+1)-Ptp1t)*Jt.';
            xtNs(:, t) = xtts(:, t) + Jt*(xtNs(:, t+1)-Abar*xtts(:, t)-Bbar*u(:, t)-SR*y(:, t));
            if t<N-1
                MtNs(:, :, t+1) = MtNs(:, :, t+1)*Jt.';
            end
            if t > 1
                MtNs(:, :, t) = Ptt+Jt*(MtNs(:, :, t+1)-Abar*Ptt);
            else
                M0N = Ptt+Jt*(MtNs(:, :, 2)-Abar*Ptt);
            end
        end
        J0 = Lambda*Abar.'/P10;
        x0N = mu+J0*(xtNs(:, 1)-Abar*mu);
        [~, R3] = qr([rootP00.'*Abar.' rootP00.'; rootQ.' zeros(n); zeros(n) rootPtNs(:, :, 1).'*J0.']);
        rootP0N = R3(n+(1:n), n+1:end).';
        P0N = rootP0N*rootP0N.';
        M0N = M0N*J0.';
        
        Phi = zeros(n);
        Psi1 = zeros(n, n+nu);
        Sigma = zeros(n+nu);
        Theta = zeros(ny);
        Psi2 = zeros(ny, n);
        for t = 1:N
            Phi = Phi + PtNs(:, :, t)+xtNs(:, t)*xtNs(:, t).';
            Psi2 = Psi2 + y(:, t)*xtNs(:, t).';
            Theta = Theta + y(:, t)*y(:, t).';
            if t>1
                Psi1 = Psi1 + [xtNs(:, t)*xtNs(:, t-1).'+MtNs(:, :, t) xtNs(:, t)*u(:, t-1).'];
                Sigma = Sigma + [xtNs(:, t-1)*xtNs(:, t-1).'+PtNs(:, :, t-1) xtNs(:, t-1)*u(:, t-1).';u(:, t-1)*xtNs(:, t-1).' u(:, t-1)*u(:, t-1).'];
            end
        end
        Psi1 = Psi1 + [xtNs(:, 1)*x0N.'+M0N zeros(n, nu)];
        Sigma = Sigma + [x0N*x0N.'+P0N zeros(n, nu);zeros(nu, n+nu)];
        
        prev = ML21(A, B, C, Q, R, mu, Lambda, x0N, P0N, Phi, Psi1, Psi2, Sigma, Theta, N);
        
        mu = x0N;
        Lambda = P0N;
        tmp = KroneckerProduct(Psi1, eye(n))/KroneckerProduct(Sigma, eye(n));
        tmp2 = Psi1/Sigma.';
        Gamma1 = Psi1/Sigma;
        A = Gamma1(:, 1:n);
        B = Gamma1(:, n+1:end);
%         R4 = robustCholensky2([Sigma Psi1.';Psi1 Phi]/N, n+nu);
%         r22 = R4(n+nu+1:end, n+nu+1:end);
%         Q = (r22.'*r22);
%         Qtest = Phi-Psi1/Sigma*Psi1.';
        C = Psi2/Phi;
%         R4 = robustCholensky2([Phi Psi2.';Psi2 Theta]/N, n);
%         r22 = R4(n+1:end, n+1:end);
%         R = (r22.'*r22);
        
        L = ML21(A, B, C, Q, R, mu, Lambda, x0N, P0N, Phi, Psi1, Psi2, Sigma, Theta, N);
        
        tries = tries + 1;
        %fprintf("\t(%u:%u) New likelihood: %.3e (changed from %.3e; relative change of %.3e)\n", mcIt, it, L, prev, abs((L-prev)/L));
        if (L-prev)/L < 5e-4
            %break
        end
        it = it+1;
        tries = tries + 1;
    end
end


function [L] = ML21(A, B, C, Q, R, mu, Lambda, x0N, P0N, Phi, Psi1, Psi2, Sigma, Theta, N)
    Gamma1 = [A B];
    L = log(det(Lambda))+sum(diag(Lambda/(x0N*x0N.'+P0N-x0N*mu.'-mu*x0N.'+mu*mu.')));
    L = L + (N*log(det(R))+sum(diag(R\(Theta-C*Psi2.'-Psi2*C.'+C*Phi*C.')))+N*log(det(Q))+sum(diag(Q\(Phi-Gamma1*Psi1.'-Psi1*Gamma1.'+Gamma1*Sigma*Gamma1.'))));
    L = -0.5*L;
end

function [R] = robustCholensky2(A, r)
    % From Golub, G., & Loan, C. V. (1989). Matrix computations. Baltimore, MD:Johns Hopkins University Press.
    % Algorithm 4.2.4 p.170
    % A = [A11 A12; A21 A22] with A11 an rxr matrix
    e = eig(A);
    me = min(e);
    if me<-1e-1
        fprintf("Robust cholensky: eigenvalue too negative: %.3e\n", me)
    end
    if me < 0
        %oldA = A;
        [V, D] = eig(A);
        D(e<0) = 0;
        A = real(V*D*V.');
    end
    
    R = zeros(size(A));
    S = A(1:r, 1:r);
    R11 = chol(S);
    R(1:r, 1:r) = R11.';
    S = A(r+1:end, 1:r);
    R21 = S/R11;
    R(r+1:end, 1:r) = R21;
    S = A(r+1:end, r+1:end)-R21*R21.';
    e = eig(S);
    me = min(e);
    assert(me>-0.1)
    if me < 0
        %oldA = A;
        [V, D] = eig(S);
        D(e<0) = abs(D(e<0));
        S = real(V*D*V.');
    end
    R(r+1:end, r+1:end) = robustCholensky(S).';
    
    R=R.';
end

function [R] = robustCholensky(A)
    % From Golub, G., & Loan, C. V. (1989). Matrix computations. Baltimore, MD:Johns Hopkins University Press.
    % Algorithm 4.2.4 p.170
    R = zeros(size(A));
    for j=1:size(A, 2)
        for i=j:size(A, 1)
            S = A(i, j) - R(i, 1:j-1)*R(j, 1:j-1).';
            if i==j
                %assert(S>-1e-4)
                if S > 0
                    R(j, j) = sqrt(S);  % Only negative numbers are very small
                else
                    fprintf("");
                end
            else
                R(i, j) = S/R(j, j);
            end
        end
    end
    R = R.';
end
