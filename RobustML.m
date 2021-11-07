function [A, B, C, D, Q, R, S, mu, Lambda] = RobustML(A, B, C, D, Q, R, S, mu, Lambda, y, u)
    
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
        %P0N2 = Lambda+J0*(PtNs(:, :, 1)-P10)*J0.';
%         xtNs(:, 1) = mu;                    %] By definition mu is the exact x1, independent from any predictions
%         rootPtNs(:, :, 1) = zeros(size(A)); %]
        
        %% E step
%         Phi_base = zeros(n);
%         Psi1_base = zeros(n, n+nu);
%         Sigma_base = zeros(n+nu);
%         Theta = zeros(ny);
%         Psi2_base = zeros(ny, n);
%         for t=1:N
%             if t>1
%                 % Complete by adding mu*mu.'
%                 Phi_base = Phi_base + xtNs(:, t)*xtNs(:, t).'+PtNs(:, :, t);
%                 % Complete by adding y(:, 1)*mu.'
%                 Psi2_base = Psi2_base + y(:, t)*xtNs(:, t).';
%             end
%             if t>2
%                 % Complete by adding [xtNs(:, 2)*mu.'+MtNs(2) xtNs(:, 2)*u(:, 1).']
%                 Psi1_base = Psi1_base + [xtNs(:, t)*xtNs(:, t-1).'+MtNs(:, :, t) xtNs(:, t)*u(:, t-1).'];
%                 % Complete by adding [mu*mu.' mu*u(:, 1).'; u(:, 1)*mu.' u(:, 1)*u(:, 1).']
%                 Sigma_base = Sigma_base + [xtNs(:, t-1)*xtNs(:, t-1).'+PtNs(:, :, t-1) xtNs(:, t-1)*u(:, t-1).';u(:, t-1)*xtNs(:, t-1).' u(:, t-1)*u(:, t-1).'];
%             end
%             Theta = Theta + y(:, t)*y(:, t).';
%         end
%         Phi = Phi_base+mu*mu.';
%         Psi1 = Psi1_base+[xtNs(:, 2)*mu.'+MtNs(:, :, 2) xtNs(:, 2)*u(:, 1).'];
%         Psi2 = Psi2_base+y(:, 1)*mu.';
%         Sigma = Sigma_base + [mu*mu.' mu*u(:, 1).'; u(:, 1)*mu.' u(:, 1)*u(:, 1).'];
%         prev = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%         if prev<0
%             fprintf("")
%         end
%         %prev2 = ML2(A, B, C, Q, R, xtNs, PtNs, MtNs, u, y);
%         L = -inf;
%         prevMu = mu;
%         while L < prev
%             %mu = (C.'/R*C+A.'/Q*A)\(C.'/R*y(:, 1)+A.'/Q*(xtNs(:, 2)-B*u(:, 1)));
%             Lmu = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             xtNs(:, 1) = mu;
%             Phi = Phi_base+mu*mu.';
%             Psi1 = Psi1_base+[xtNs(:, 2)*mu.'+MtNs(:, :, 2) xtNs(:, 2)*u(:, 1).'];
%             Psi2 = Psi2_base+y(:, 1)*mu.';
%             Sigma = Sigma_base + [mu*mu.' mu*u(:, 1).'; u(:, 1)*mu.' u(:, 1)*u(:, 1).'];
%             
%             Gamma1 = Psi1/Sigma;
%             assert(max(abs(imag(Gamma1)), [], 'all') == 0);
% %     xttm1s = robustKalmanForward(A, B, C, Q, R, mu, uval, yval);
% %     sc1 = predictionErrorCost(C, xttm1s, yval_noiseless);
% %             tst = Gamma1(:, 1:n)-A;
%             A = Gamma1(:, 1:n);
%             La = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
% %             La2 = ML(A+1.05*tst, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
% %             La3 = ML(A-0.95*tst, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             La2 = ML2(A, B, C, Q, R, xtNs, PtNs, MtNs, u, y);
% %     xttm1s = robustKalmanForward(A, B, C, Q, R, mu, uval, yval);
% %     sc2 = predictionErrorCost(C, xttm1s, yval_noiseless);
%             B = Gamma1(:, n+1:end);
%             %Lb = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             R4 = robustCholensky2([Sigma Psi1.';Psi1 Phi_base]/N, n+nu);
%             r22 = R4(n+nu+1:end, n+nu+1:end);
%             Q = (r22.'*r22);
%             %assert(max(Q,[],'all')<100); 
%             %Lq = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             %Q = (Phi_base-Gamma1*Psi1.')/N;
%             C = Psi2/Phi;
%             %Lc = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             R4 = robustCholensky2([Phi Psi2.';Psi2 Theta]/N, n);
%             r22 = R4(n+1:end, n+1:end);
%             R = (r22.'*r22);
%             %Lr = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             %R = (Theta-C*Psi2.')/N;
%             L = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N);
%             assert(abs(imag(L)) == 0)
%             %assert(L > prev)
%             %L2 = ML2(A, B, C, Q, R, xtNs, PtNs, MtNs, u, y);
%             if max(abs(prevMu-mu))<1e-6
%                 %break
%             end
%             prevMu = mu;
%         end

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
        Gamma1 = Psi1/Sigma;
        A = Gamma1(:, 1:n);
        B = Gamma1(:, n+1:end);
        R4 = robustCholensky2([Sigma Psi1.';Psi1 Phi]/N, n+nu);
        r22 = R4(n+nu+1:end, n+nu+1:end);
        Q = (r22.'*r22);
        Qtest = Phi-Psi1/Sigma*Psi1.';
        C = Psi2/Phi;
        R4 = robustCholensky2([Phi Psi2.';Psi2 Theta]/N, n);
        r22 = R4(n+1:end, n+1:end);
        R = (r22.'*r22);
        
        L = ML21(A, B, C, Q, R, mu, Lambda, x0N, P0N, Phi, Psi1, Psi2, Sigma, Theta, N);
        
%         Phi = zeros(n+ny);
%         Psi = zeros(n+ny, n+nu);
%         Sigma = zeros(n+nu);
%         
%         for t=2:N
%             Phi = Phi+[xtNs(:, t)*xtNs(:, t).'+PtNs(:, :, t) xtNs(:, t)*y(:, t-1).';y(:, t-1)*xtNs(:, t).' y(:, t-1)*y(:, t-1).'];
%             Psi = Psi+[xtNs(:, t)*xtNs(:, t-1).'+MtNs(:, :, t) xtNs(:, t)*u(:, t-1).';y(:, t-1)*xtNs(:, t-1).' y(:, t-1)*u(:, t-1).'];
%             Sigma = Sigma+[xtNs(:, t-1)*xtNs(:, t-1).'+PtNs(:, :, t-1) xtNs(:, t-1)*u(:, t-1).'; u(:, t-1)*xtNs(:, t-1).' u(:, t-1)*u(:, t-1).'];
%         end
%         y0 = C*x0N;
%         Phi = Phi+[xtNs(:, 1)*xtNs(:, 1).'+PtNs(:, :, 1) xtNs(:, t)*y0.';y0*xtNs(:, t).' y0*y0.'];
%         Psi = Psi+[xtNs(:, 1)*x0N.'+M0N zeros(n, nu);y0*x0N.' zeros(ny, nu)];
%         Sigma = Sigma+[x0N*x0N.'+P0N zeros(n, nu); zeros(nu, n+nu)];
%         prev = ML3([A B;C D], [Q S;S.' R], Phi, Psi, Sigma, x0N, P0N, mu, Lambda, N);
%         
%         mu = x0N;
%         Lambda = P0N;
%         Gamma = Psi/Sigma;
%         A = Gamma(1:n, 1:n);
%         B = Gamma(1:n, n+1:end);
%         C = Gamma(n+1:end, 1:n);
%         D = Gamma(n+1:end, n+1:end);
%         
%         R4 = robustCholensky2([Sigma Psi.';Psi Phi]/N, n+nu);
%         r22 = R4(n+nu+1:end, n+nu+1:end);
%         Pi2 = (r22.'*r22);
%         Pi = (Phi-Psi/Sigma*Psi.')/N;
%         Q = Pi(1:n, 1:n);
%         S = Pi(1:n, n+1:end);
%         S = zeros(size(S));
%         R = Pi(n+1:end, n+1:end);
%         
%         L = ML3(Gamma, Pi, Phi, Psi, Sigma, x0N, P0N, mu, Lambda, N);
        
        assert(norm(Q-Q.', 'fro') < 1e-10);
        assert(max(eig(Q))>0);
        assert(norm(R-R.', 'fro') < 1e-10);
        assert(max(eig(R))>0);
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
