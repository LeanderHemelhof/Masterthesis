clear
close all
clc
rng(698614648)

% Aref = 0.5;
% Bref = 2.83;
% Cref = 1.2;
% Qref = [0.01];
% Rref = [0.25];
% Sref = 0;
% muref = 16;
% P1ref = 0.5;

Aref = [0.5 0;0 0.183];
Bref = [2.83 1;-0.5 8];
Cref = [1.2 1; -1 1;5 2];

n4opts = n4sidOptions('Focus', 'prediction', 'N4Weight', 'CVA');
ref = drss(5, 2, 2);
Aref = ref.A;
Bref = ref.B;
Cref = ref.C;
Dref = ref.D;
Ts = 1/50;

Qref = eye(size(Aref, 1))*0.04;%[0.01 0;0 0.03];
Rref = eye(size(Cref, 1))*0.04;%[0.25 0 0; 0 0.12 0; 0 0 0.6];
Sref = zeros(size(Qref, 1), size(Rref, 2));
muref = zeros(size(Aref, 1), 1);%[16; -12];
P1ref = [0.5 0; 0 0.5];

N = 500;
u = randn(size(Bref, 2), N);%rand(size(Bref, 2), N)*5;
assert(min(eig(u*u.'))>0)
uval = randn(size(Bref, 2), N);%rand(size(Bref, 2), N)*5;
assert(min(eig(uval*uval.'))>0)


nx = size(Aref, 2);
ny = size(Cref, 1);
nu = size(Bref, 2);
s = zeros(nu);
for t=1:N
    s = s + u(:, t)*u(:, t).';
end
assert(min(eig(s)) > 0);


mcCnt = 100;
scores = zeros(1, mcCnt);
predScores = zeros(1, mcCnt);
referencePredScores = zeros(1, mcCnt);
nAs = zeros([size(Aref) mcCnt]);
nBs = zeros([size(Bref) mcCnt]);
nCs = zeros([size(Cref) mcCnt]);
nDs = zeros([ny nu mcCnt]);
nQs = zeros([size(Qref) mcCnt]);
nRs = zeros([size(Rref) mcCnt]);
nSs = zeros([nx ny mcCnt]);
nmus = zeros([size(muref) mcCnt]);
nP1s = zeros([size(P1ref) mcCnt]);

mainTic = tic();
for mcIt = 1:mcCnt
    y = zeros(ny, N);
    x = muref;%+sqrt(P1ref)*randn(nx, 1);
    for i=1:N
        y(:, i) = Cref*x+sqrt(Rref)*randn(ny, 1);
        x = Aref*x+Bref*u(:, i)+sqrt(Qref)*randn(nx, 1);
    end
    
    %% Generate validation realisation using real system
    yval = zeros(ny, N);
    x = muref;%+sqrt(P1ref)*randn(nx, 1);
    for i=1:N
        yval(:, i) = Cref*x+sqrt(Rref)*randn(ny, 1);
        x = Aref*x+Bref*uval(:, i)+sqrt(Qref)*randn(nx, 1);
    end
    dat = iddata(y.', u.', Ts);
    n = size(Aref, 1);
    sys = n4sid(dat, n, n4opts);
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;
    Q = eye(nx);
    R = eye(ny)*0.2;
    S = zeros(nx, ny);
    %[A, B, C, D, Q, R, S] = SSEstim(y, u, n);
    %mu = pinv(C)*y(:, 1);
    Ys = [];
    Us = [];
    O = [];
    F = [];
    nn = n;%max(10, n);
    for t=1:nn
        Ys = [Ys;y(:, t)];
        Us = [Us; u(:, t)];
        O = [O;C*A^(t-1)];
        tmp = zeros(ny, nu*(nn-t+1));
        for i=t-1:-1:1
            tmp = [C*A^(t-i-1)*B tmp];
        end
        F = [F;tmp];
    end
    % Ys=O*mu+F*Us => (Ys-F*Us)=O*mu
    mu = O\(Ys-F*Us);
    P1 = eye(size(A, 1));%(rand(size(P1ref))/0.15+0.25).*P1ref;
    
    SR = S/R;
    %SR = zeros(size(SR));
    Abar = A-SR*C;
    Bbar = B-SR*D;
    Qbar = Q-SR*S.';
    xttm1s = robustKalmanForward(Abar, Bbar, C, D, Qbar, R, SR, mu, P1, uval, yval);
    referencePredScores(mcIt) = predictionErrorCost(A, B, C, D, mu, xttm1s, uval, yval);
    
%     mu = (rand(size(muref))/0.15+0.25).*muref;
%     T = pinv(C)*Cref;
%     tA = T\A*T;
%     tB = T\B;
%     tC = C*T;
%     tmu = T\mu;
    
%     A = (rand(size(Aref))/0.15+0.25).*Aref;
%     B = (rand(size(Bref))/0.15+0.25).*Bref;
%     C = (rand(size(Cref))/0.15+0.25).*Cref;
%     D = zeros(ny, nu);
%     Q = (rand(size(Qref))/0.15+0.25).*Qref;
    %R = Rref;
    %S = Sref;
    
    it = 1;
    innerTic = tic();
    tries = 0;
    while tries ~= 100%< 1000
        %% Transform matrices to remove covariance of Pi
        SR = S/R;
        %SR = zeros(size(SR));
        Abar = A-SR*C;
        Bbar = B-SR*D;
        Qbar = Q-SR*S.';
        
        %% Initialize Kalman recursion
        rootPtts = zeros([size(P1) N]);
        %rootPtts(:, :, 1) = sqrtm(P1);
        rootPttm1 = sqrtm(P1);
        xtts = zeros(size(mu, 1), N);
        xttm1s = zeros(size(mu, 1), N);
        %xtts(:, 1) = mu;
        xttm1s(:, 1) = mu;
        xttm1 = mu;
        
        %% Robust forward
        if mcIt == 5
            fprintf("");
        end
        rootQbar = sqrtm(Qbar);
        rootR = sqrtm(R);
        rootPtp1ts = zeros([size(P1) N-1]);
        for t=1:N
            if t>1
                [~, R1] = qr([(Abar*rootPtts(:, :, t-1)).';rootQbar.']);
                rootPttm1 = R1(1:nx, :).';
                rootPtp1ts(:, :, t-1) = rootPttm1;
                xttm1 = Abar*xtts(:, t-1)+Bbar*u(:, t-1)+SR*y(:, t-1);
            end
            [~, R2] = qr([rootR, C*rootPttm1;zeros(nx, ny), rootPttm1].');
            rootPtts(:, :, t) = R2(ny+1:end, ny+1:end).';
            
            Pttm1 = rootPttm1*rootPttm1.';
            Kt = Pttm1*C.'/(C*Pttm1*C.'+R);
            xtts(:, t) = xttm1+Kt*(y(:, t)-C*xttm1-D*u(:, t));
            xttm1s(:, t) = xttm1;
        end
        
        %% Robust backward
        rootPtNs = zeros(size(rootPtts));
        PtNs = zeros([size(P1) N+1]);
        xtNs = zeros(size(mu, 1), N+1);
        MtNs = zeros([size(A) N+1]);
        
        rootPtNs(:, :, N) = rootPtts(:, :, N);
        PtNs(:, :, N) = rootPtts(:, :, N)*rootPtts(:, :, N).';
        [~, R1] = qr([(Abar*rootPtts(:, :, N)).';rootQbar.']);
        rootPNp1N = R1(1:nx, :).';
        PtNs(:, :, N+1) = rootPNp1N*rootPNp1N.';
        xtNs(:, N+1) = Abar*xtts(:, N)+Bbar*u(:, N)+SR*y(:, N);
        xtNs(:, N) = xtts(:, N);
        %KNp1 = PtNs(:, :, N+1)*C.'/(C*PtNs(:, :, N+1)*C.'+R);
        MtNs(:, :, N+1) = zeros(size(A));%(eye(nx)-KNp1*C)*Abar*PtNs(:, :, N);
        MtNs(:, :, N) = (eye(nx)-Kt*C)*Abar*(rootPtts(:, :, N-1)*rootPtts(:, :, N-1).');
        for t=N-1:-1:1
            rootPtt = rootPtts(:, :, t);
            Ptt = rootPtt*rootPtt.';
            rootPtp1t = rootPtp1ts(:, :, t);
            rootPtp1N = rootPtNs(:, :, t+1);
            Jt = Ptt*Abar.'/(rootPtp1t*rootPtp1t.');
            [~, R3] = qr([rootPtt.'*Abar.' rootPtt.'; rootQbar.' zeros(nx); zeros(nx) rootPtp1N*Jt.']);
            rootPtNs(:, :, t) = R3(nx+(1:nx), nx+1:end);
            
%             test = Ptt + Jt*((rootPtp1N.'*rootPtp1N)-(rootPtp1t*rootPtp1t.'))*Jt.';
%             test2 = Ptt+Jt*((rootPtp1N.'*rootPtp1N))*Jt.'-Ptt*Abar.'/((rootPtp1t*rootPtp1t.'))*Abar*Ptt;
            
            PtNs(:, :, t) = rootPtNs(:, :, t).'*rootPtNs(:, :, t);
            xtNs(:, t) = xtts(:, t) + Jt*(xtNs(:, t+1)-Abar*xtts(:, t)-Bbar*u(:, t)-SR*y(:, t));
            if t<N-1
                MtNs(:, :, t+1) = MtNs(:, :, t+1)*Jt.';
            end
            if t > 1
                MtNs(:, :, t) = Ptt+Jt*(MtNs(:, :, t+1)-Abar*Ptt);
            end
        end
        
        %% E step
        Phi = zeros(nx+ny);
        Psi = zeros(nx+ny, nx+nu);
        Sigma = zeros(nx+nu);
%         Phi2 = zeros(nx+ny);
%         Psi2 = zeros(nx+ny, nx+nu);
%         Sigma2 = zeros(nx+nu);
        for t = 1:N
            Phi = Phi + [xtNs(:, t+1)*xtNs(:, t+1).'+PtNs(:, :, t+1) xtNs(:, t+1)*y(:, t).';y(:, t)*xtNs(:, t+1).' y(:, t)*y(:, t).'];
            Psi = Psi + [xtNs(:, t+1)*xtNs(:, t).'+MtNs(:, :, t+1) xtNs(:, t+1)*u(:, t).'; y(:, t)*xtNs(:, t).' y(:, t)*u(:, t).'];
            Sigma = Sigma + [xtNs(:, t)*xtNs(:, t).'+PtNs(:, :, t) xtNs(:, t)*u(:, t).'; u(:, t)*xtNs(:, t).' u(:, t)*u(:, t).'];
%             Phi2 = Phi2 + [xtNs(:, t+1)*xtNs(:, t+1).'+PtNs(:, :, t+1) xtNs(:, t+1)*y(:, t).';y(:, t)*xtNs(:, t+1).' y(:, t)*y(:, t).'];
%             Psi2 = Psi2 + [xtNs(:, t+1)*xtNs(:, t).'+MtNs(:, :, t+1) xtNs(:, t+1)*u(:, t).'; y(:, t)*xtNs(:, t).' y(:, t)*u(:, t).'];
%             Sigma2 = Sigma2 + [xtNs(:, t)*xtNs(:, t).'+PtNs(:, :, t) xtNs(:, t)*u(:, t).'; u(:, t)*xtNs(:, t).' u(:, t)*u(:, t).'];
        end
        Phi = Phi/N;
        Psi = Psi/N;
        Sigma = Sigma/N;
%         Phi2 = Phi2/N;
%         Psi2 = Psi2/N;
%         Sigma2 = Sigma2/N;
        assert(min(eig(Sigma))>0)
        assert(min(eig(Phi))>0)
        prev = ML([A B;C D], [Q S;S.' R], Phi, Psi, Sigma, P1, mu, xtNs(:, 1), N);
        
        %% Robust M step
        mu = xtNs(:, 1);
        P1 = PtNs(:, :, 1);
        Gamma = Psi/Sigma;
%         %R4 = chol([Sigma Psi.';Psi Phi]);
%         [U, d] = eig([Sigma Psi.';Psi Phi], 'vector'); % A == U*diag(d)*U'
%         d(d < 0) = 0; % Set negative eigenvalues of A to zero
%         [~, R4t] = qr(diag(sqrt(d))*U'); % Q*R == sqrt(D)*U', so A == U*diag(d)*U'
%                                         %                         == R'*Q'*Q*R == R'*R
%         R42 = robustCholensky([Sigma Psi.';Psi Phi]);
        assert(min(eig([Phi Psi;Psi.' Sigma]))>=0)
        R4 = robustCholensky2([Sigma Psi.';Psi Phi], nx+nu);
        assert(norm(R4.'*R4-[Sigma Psi.';Psi Phi], 'fro') < 100);
%         %assert(norm(R42.'*R42-[Sigma Psi.';Psi Phi], 'fro') < 100);
        r22 = R4(nx+nu+1:end, nx+nu+1:end);
        Pi = r22.'*r22;
        %Pi = Phi-Psi/Sigma*Psi.';
        A = Gamma(1:nx, 1:nx);
        B = Gamma(1:nx, nx+1:end);
        C = Gamma(1+nx:end, 1:nx);
        D = Gamma(1+nx:end, 1+nx:end);
        
        Q = Pi(1:nx, 1:nx);
        S = Pi(1:nx, nx+1:end);
        R = Pi(1+nx:end, 1+nx:end);
        L = ML(Gamma, Pi, Phi, Psi, Sigma, P1, mu, xtNs(:, 1), N);
        %fprintf("\t(%u:%u) New likelihood: %.3e (changed from %.3e; relative change of %.3e)\n", mcIt, it, L, prev, abs((L-prev)/L));
        if abs((L-prev)/L) < 5e-4
            break
        end
        it = it+1;
        tries = tries + 1;
    end
    fprintf("MC run %u done. Took %.3fs\n", mcIt, toc(innerTic))
    scores(mcIt) = L;
    SR = S/R;
    %SR = zeros(size(SR));
    Abar = A-SR*C;
    Bbar = B-SR*D;
    Qbar = Q-SR*S.';
    xttm1s = robustKalmanForward(Abar, Bbar, C, D, Qbar, R, SR, mu, P1, uval, yval);
    predScores(mcIt) = predictionErrorCost(A, B, C, D, mu, xttm1s, uval, yval);
    T = pinv(C)*Cref;
    nAs(:, :, mcIt) = T\A*T;
    nBs(:, :, mcIt) = T\B;
    nCs(:, :, mcIt) = C*T;
    nDs(:, :, mcIt) = D;
    nQs(:, :, mcIt) = T\Q*inv(T).';
    nRs(:, :, mcIt) = R;
    nSs(:, :, mcIt) = S;
    nmus(:, :, mcIt) = T\mu;
    nP1s(:, :, mcIt) = P1;
end
fprintf("All MC runs done. Took %.3fs\n", toc(mainTic))
figure
boxplot(scores)
figure
boxplot(predScores)
figure
v = max([predScores referencePredScores]);
plot([0 v], [0 v], 'b-')
cols = 'rg';
ids = ones(size(predScores));
ids(referencePredScores>predScores) = 2;
hold on
for i=1:mcCnt
    plot(referencePredScores(i), predScores(i), [cols(ids(i)) '*'])
end
fprintf("EM better: %u; SS better: %u\n", sum(referencePredScores>predScores), sum(referencePredScores<predScores))
fprintf("Values +- 2*sigma (95 percent confidence interval)\n")
W = lerp(0.05, 0.95, min(-predScores), max(-predScores), -predScores);
wm = weightedMean(nAs, W);
wv = weightedVariance(nAs, wm, W);
printMatWithStd('A', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nBs, W);
wv = weightedVariance(nBs, wm, W);
printMatWithStd('B', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nCs, W);
wv = weightedVariance(nCs, wm, W);
printMatWithStd('C', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nDs, W);
wv = weightedVariance(nDs, wm, W);
printMatWithStd('D', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nQs, W);
wv = weightedVariance(nQs, wm, W);
printMatWithStd('Q', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nRs, W);
wv = weightedVariance(nRs, wm, W);
printMatWithStd('R', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nSs, W);
wv = weightedVariance(nSs, wm, W);
printMatWithStd('S', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nmus, W);
wv = weightedVariance(nmus, wm, W);
printMatWithStd('mu', wm, 2*sqrt(wv))
fprintf("\n");
wm = weightedMean(nP1s, W);
wv = weightedVariance(nP1s, wm, W);
printMatWithStd('P1', wm, 2*sqrt(wv))
fprintf("\n");
%fprintf("Prediction error cost of mean: %.3f\n", predictionErrorCost(weightedMean(nAs, W), weightedMean(nBs, W), weightedMean(nCs, W), weightedMean(nDs, W), weightedMean(nmus, W), uval, yval))

function [L] = ML(Gamma, Pi, Phi, Psi, Sigma, P1, mu, x1N, N)
    Exmu = (x1N-mu)*(x1N-mu).'+P1;
    L = -0.5*(log(det(P1))+sum(diag(P1\Exmu)));
    L = L-0.5*(N*log(det(Pi))+N*sum(diag(Pi\(Phi-Psi*Gamma.'-Gamma*Psi.'+Gamma*Sigma*Gamma.'))));
end

function [v] = lerp(low, high, b1, b2, val)
    tmp = (val-b1)/(b2-b1);
    v = low+tmp*(high-low);
end

function [m] = weightedMean(Ms, W)
    m = zeros(size(Ms, [1 2]));
    for i=1:size(Ms, 3)
        m = m+Ms(:, :, i)*W(i);
    end
    m = m/sum(W);
end

function [m] = weightedVariance(Ms, wm, W)
    m = zeros(size(Ms, [1 2]));
    for i=1:size(Ms, 3)
        m = m+W(i)*(Ms(:, :, i)-wm).^2;
    end
    m = m/sum(W);
end

function [cost] = predictionErrorCost(A, B, C, D, mu, xttm1s, u, y)
    cost = 0;
    %x = mu;
    for t = 1:size(y, 2)        
        tmp = y(:, t)-C*(xttm1s(:, t))+D*u(:, t);
        %tmp = y(:, t)-C*x+D*u(:, t);
        cost = cost + tmp.'*tmp;
        %x = A*x+B*u(:, t);
    end
    cost = cost/numel(y);
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

function [R] = robustCholensky2(A, r)
    % From Golub, G., & Loan, C. V. (1989). Matrix computations. Baltimore, MD:Johns Hopkins University Press.
    % Algorithm 4.2.4 p.170
    % A = [A11 A12; A21 A22] with A11 an rxr matrix
    R = zeros(size(A));
    S = A(1:r, 1:r);
    R11 = chol(S);
    R(1:r, 1:r) = R11.';
    S = A(r+1:end, 1:r);
    R21 = S/R11;
    R(r+1:end, 1:r) = R21;
    S = A(r+1:end, r+1:end)-R21*R21.';
    R(r+1:end, r+1:end) = chol(S).';
    
    R=R.';
end

function [xttm1s] = robustKalmanForward(Abar, Bbar, C, D, Qbar, R, SR, mu, P1, u, y)
    N = size(u, 2);
    nx = size(Abar, 1);
    ny = size(y, 1);
    rootPtts = zeros([size(P1) N]);
    rootPtts(:, :, 1) = sqrtm(P1);
    xtts = zeros(size(mu, 1), N);
    xttm1s = zeros(size(mu, 1), N);
    xtts(:, 1) = mu;
    xttm1s(:, 1) = mu;

    rootQbar = sqrtm(Qbar);
    rootR = sqrtm(R);
    rootPtp1ts = zeros([size(P1) N-1]);
    for t=2:N
        [~, R1] = qr([(Abar*rootPtts(:, :, t-1)).';rootQbar.']);
        rootPttm1 = R1(1:nx, :).';
        rootPtp1ts(:, :, t-1) = rootPttm1;
        [~, R2] = qr([rootR, C*rootPttm1;zeros(nx, ny), rootPttm1].');
        rootPtts(:, :, t) = R2(ny+1:end, ny+1:end).';

        xttm1 = Abar*xtts(:, t-1)+Bbar*u(:, t-1)+SR*y(:, t-1);
        Pttm1 = rootPttm1.'*rootPttm1;
        Kt = Pttm1*C.'/(C*Pttm1*C.'+R);
        xtts(:, t) = xttm1+Kt*(y(:, t)-C*xttm1-D*u(:, t));
        xttm1s(:, t) = xttm1;
    end
end

function [] = printMatWithStd(name, A, stdMat)
    rtp = ceil(size(A, 1)/2);
    l = length(name)+3;
    for row=1:size(A, 1)
        if row == rtp
            fprintf("%s = ", name);
        else
            fprintf(repmat(' ', 1, l));
        end
        for j=1:size(A, 2)
            fprintf('%.3f+-%.3f ', A(row, j), stdMat(row, j));
        end
        %fprintf(repmat('%.3f+-%.3f ', 1, size(A, 2)), A(row, :), stdMat(row, :));
        fprintf("\n")
    end
end