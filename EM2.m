addpath('./utils')

clear
close all
clc
rng(345178642)

% Aref = 0.5;
% Bref = 2.83;
% Cref = 1.2;
% Qref = [0.01];
% Rref = [0.25];
% Sref = 0;
% muref = 16;
% P1ref = 0.5;

fs = 50;
Aref = [1 1/fs 0;0 1 1/fs;0 -0.1 0];
Bref = [0;0;1/50];
Cref = [1/0.17 0 0];
Dref = [0];
refModel = ss(Aref, Bref, Cref, Dref, 1/fs);
n = 3;

% Aref = [0.5 0;0 0.183];
% Bref = [2.83 1;-0.5 8];
% Cref = [1.2 1; -1 1;5 2];
% n = 5;

% n4opts = n4sidOptions('Focus', 'prediction', 'N4Weight', 'CVA');
ref = drss(n, 1, 1);
Aref = ref.A;
Bref = ref.B;
Cref = ref.C;
Dref = 0*ref.D;
Ts = 1/50;

ynn = zeros(size(Cref, 1), 500);
unn = randn(size(Bref, 2), 500);
x = [0;0;0];%[16;-12;6;8;3.7];
for t=1:500
    ynn(:, t) = Cref*x;
    x = Aref*x+Bref*unn(:, t);
end
%[A, B, C, D, ~, ~, ~] = SSEstim(ynn, unn, n);

O = Cref;
for t=1:n-1
    O = [O;Cref*Aref^t];
end
assert(rank(O) == n);

Qref = eye(size(Aref, 1))*0.004;%[0.01 0;0 0.03];
Rref = eye(size(Cref, 1))*0.04;%[0.25 0 0; 0 0.12 0; 0 0 0.6];
muref = [13;4;2];%[16;-12;6;8;3.7];%zeros(size(Aref, 1), 1);
[Kref, ~] = findKandR(Aref, Cref, Qref, Rref, zeros(size(Qref, 1), size(Rref, 2)));

N = 1000;
u = filteredNoise(size(Bref, 2), N, 2, 0, fs);%5, fs);%randn(size(Bref, 2), N)*2;%rand(size(Bref, 2), N)*5;
% u2 = zeros(size(Bref, 2), N);
% u2(1) = 1;
[u2, bins] = CreateMultisine(fs/N, (fs/2-fs/N), N, fs, 'schroeder', 2.2);
%u2 = sin(2*pi*fs/N*(1:N)+512*513*pi/2499)+sin(2*pi*(fs/2-fs/N)*(1:N)+513*514*pi/2499);
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
nmus = zeros([size(muref) mcCnt]);

bestScore = inf;
bestOutput = [];
mainTic = tic();
for mcIt = 1:mcCnt
    y = zeros(ny, N);
    initX = Aref*muref;%+mvnrnd(zeros(1, nx), Qref).';
    x = initX;
    xs = x;
    for i=1:N
        y(:, i) = Cref*x+mvnrnd(zeros(1, ny), 20*Rref).';%+sqrt(Rref)*randn(ny, 1);
        x = Aref*x+Bref*u(:, i);%+mvnrnd(zeros(1, nx), Qref).';
        xs = [xs x];
    end
    
    y2 = zeros(ny, N);
    initX2 = zeros(nx, 1);%Aref*muref+mvnrnd(zeros(1, nx), Qref).';
    x2 = initX2;
    xs2 = x2;
    for i=1:N
        y2(:, i) = Cref*x2+mvnrnd(zeros(1, ny), Rref).';%+sqrt(Rref)*randn(ny, 1);
        x2 = Aref*x2+Bref*u2(:, i)+mvnrnd(zeros(1, nx), Qref).';
        xs2 = [xs2 x2];
    end
    
    y2fil = zeros(ny, N);
    x2=initX2;
    for i=1:N
        y2fil(:, i) = Cref*x2+mvnrnd(zeros(1, ny), Rref).';%+sqrt(Rref)*randn(ny, 1);
        e = y2fil(:, i)-(Cref*x2+Dref*u(:, i));
        x2 = Aref*x2+Bref*u2(:, i)+Kref*e;
    end
    y2nn = zeros(ny, N);
    x2 = initX2;
    for i=1:N
        y2nn(:, i) = Cref*x2+Dref*u(:, i);
        x2 = Aref*x2+Bref*u2(:, i);
    end
    %[Aa, Bb, Cc, Dd] = estimateModel(reshape(fft(y2)./fft(u2), 1, 1, []), 3, 0, 2500);
    
    %% Generate validation realisation using real system
    yval = zeros(ny, N);
    yval_noiseless = zeros(ny, N);
    x = Aref*muref+mvnrnd(zeros(1, nx), Qref).';%muref;%+sqrt(P1ref)*randn(nx, 1);
    xvals = x;
    for i=1:N
        yval_noiseless(:, i) = Cref*x;
        yval(:, i) = yval_noiseless(:, i)+sqrt(Rref)*randn(ny, 1);
        x = Aref*x+Bref*uval(:, i)+sqrt(Qref)*randn(nx, 1);
        xvals = [xvals x];
    end
    
    yc = zeros(ny, N);
    uc = [];
    xcInit = Aref*muref+mvnrnd(zeros(1, nx), Qref).';
    x = xcInit;%+sqrt(P1ref)*randn(nx, 1);
    xcs = x;
    th = 25;
    ref = [];
    for i=1:N
        yc(:, i) = Cref*x;%+sqrt(Rref)*randn(ny, 1);
        if mod(i, 500) == 0
            th = th + randn(1)*50;
        end
        ref = [ref th];
        ut = -2.11*(th - yc(:, i));
        %ut = pinv(Cref*Bref)*(th-Cref*Aref*x);%[5+30*randn(1);30*randn(1)];
        %ut(2) = 30*randn(1);
        uc = [uc ut];
        x = Aref*x+Bref*ut;%+sqrt(Qref)*randn(nx, 1);
        xcs = [xcs x];
    end
    
    figure
    hold on
    plot(ref)
    plot(yc)
    
    dat = iddata(y.', u.', Ts);
    n = size(Aref, 1);
    %sys = n4sid(dat, n, n4opts);
%     A = sys.A;
%     B = sys.B;
%     C = sys.C;
%     D = sys.D;
    %[A, B, C, D, ~, ~, ~] = SSEstim(y, u, n);
    %[A,B,C,D,~, ~, ~] = subid(y,u,10,5,[],[],1);
    [A, B, C, D, Q, R, S, K, Rr] = SSEstim3(y, u, n);
    Q = eye(nx)*0.1;
    R = eye(ny)*0.5;
    S = zeros(nx, ny);
    D = zeros(size(D));
    Ys = [];
    Us = [];
    O = [];
    F = [];
    Us = zeros(nu, 1);
    nn = max(20, n);
    COV = zeros(nn*ny);
    for t=1:nn
        Ys = [Ys;y2(:, t)];
        Us = [Us; u2(:, t)];
        O = [O;C*A^t];
        tmp = zeros(ny, nu*(nn-t+1));
        tmpCov = zeros(ny, ny);
        for i=t-1:-1:0
            CA = C*A^(t-i-1);
            tmp = [CA*B tmp];
            tmpCov = tmpCov + CA*Q*CA.';
        end
        F = [F;tmp];
        COV(ny*(t-1)+(1:ny), ny*(t-1)+(1:ny)) = tmpCov+R;
    end
    assert(rank(O) == n);
    % Ys=O*mu+F*Us => (Ys-F*Us)=O*mu
    %mu2 = O\(Ys-F*Us);
    W = inv(COV);
    mu = (O.'*W*O)\O.'*W*(Ys-F*Us);
    Lambda = inv(O.'/COV*O);%pinv(O)*COV*pinv(O).';
    [muTmp, LambdaTmp] = estimateInitialState(A, B, C, D, u2, y2, 20, 1, Q, R);
    plotComparison(y2, u2, A, B, C, D, K, 'Comparison between measurements and Subspace based model', 1:250)
    plotTFComparison(refModel, A, B, C, D, 'TF Comparison between reference and Subspace based model')
    [m1, p1, w1] = dbode(Aref, Bref, Cref, Dref, 1/fs, 1);
    [m2, p2, w2] = dbode(A, B, C, D, 1/fs, 1);
    figure
    hold on
    plot(w1, m1);
    plot(w2, m2);
    legend('Reference TF', 'Model TF')

    [steadyStateCov, ssK] = getSSCov(A, C, Q, R);
    
    [ys, X] = dlsim(A-K*C, [B-K*D K], C, [D zeros(size(D, 1))], [uc.' yc.'], A*mu);
    ys = ys.';
    ys2 = dlsim(A, B, C, D, uc.', A*mu).';
    if max(max(abs(ys-yc), [], 1)) > max(abs(yc), [], 'all')*10
        %% System is too unstable to be a reliable estimate, stabilize it
        [V, Si] = eig(A);
        Di = diag(Si);
        ids = abs(Di)>0.9;
        Di(ids) = Di(ids)./(1.1*abs(Di(ids)));
        A = real(V*diag(Di)/V);
        xttm1s2 = robustKalmanForward(A, B, C, D, Q, R, S, mu, Lambda, uc, yc);
        fprintf("Initial estimate too unstable. Improving...\n");
    end
%     figure
%     hold on
%     plot(yc)
%     plot(sim(C, D, xcs, uc))
%     plot(sim(C, D, xttm1s2, uc))
%     plot(sim(C, D, xttm1s, uc))
%     legend("Validation output", "No prediction", "Optimal OSH prediction", "Steady State OSH prediction")
%     
%     figure
%     plot(yc)
%     figure
%     plot(sim(C, D, xvals, uc))
%     figure
%     plot(sim(C, D, xttm1s2, uc))
%     figure
%     plot(sim(C, D, xttm1s, uc))
%     
%     figure
%     hold on
%     plot(sim(C, D, xttm1s2, uc))
%     plot(sim(C, D, xttm1s, uc))
    referencePredScores(mcIt) = predictionErrorCost2(A, B, C, D, K, mu, yc, uc);
    if referencePredScores(mcIt) > 100
        fprintf("Baseline score relatively bad: %.3f\n", referencePredScores(mcIt))
    end
    
    innerTic = tic();
    [A, B, C, D, Q, R, S, mu, Lambda] = RobustML(A, B, C, D, Q, R, S, mu, Lambda, y2, u2);
    
%     xttm1s = robustKalmanForward(A, B, C, D, Q, R, S, mu, Lambda, uc, yc);
%     predScores(mcIt) = predictionErrorCost(C, D, xttm1s, yc, uc);
    [K, Rr] = findKandR(A, C, Q, R, S);
    predScores(mcIt) = predictionErrorCost2(A, B, C, D, K, mu, yc, uc);
    fprintf("MC run %u done (scored %.3f vs original score of %.3f). Took %.3fs\n", mcIt, predScores(mcIt), referencePredScores(mcIt), toc(innerTic))
    if predScores(mcIt) < bestScore
        bestScore = predScores(mcIt);
        bestOutput = dlsim(A-K*C, [B-K*D K], C, [D zeros(size(D, 1))], [uc.' yc.'], A*mu);
    end
    figure
    hold on
    plot(ref(1:250))
    plot(yc(1:250))
    plot(ys(1:250))
    plot(bestOutput(1:250))
    legend("Target output", "Noisy reference output", "Kalman filtered subspace based model", "Kalman filitered ML based model")
    figure
    hold on
    plot(ref(1:250))
    plot(yc(1:250))
    plot(ys2(1:250))
    bo2 = dlsim(A, B, C, D, uc.', A*mu).';
    plot(bo2(1:250))
    legend("Target output", "Noisy reference output", "Unfiltered subspace based model", "Unfilitered ML based model")
%     T = pinv(C)*Cref;
%     nAs(:, :, mcIt) = T\A*T;
%     nBs(:, :, mcIt) = T\B;
%     nCs(:, :, mcIt) = C*T;
%     nQs(:, :, mcIt) = T\Q*inv(T).';
%     nRs(:, :, mcIt) = R;
%     nmus(:, :, mcIt) = T\mu;
end
fprintf("Took %.3fs", toc(mainTic))
fprintf("EM better: %u; SS better: %u\n", sum(referencePredScores>predScores), sum(referencePredScores<predScores))
figure
m = max(max(predScores), max(referencePredScores));
plot([0 m], [0 m], 'b-')
cols = 'rg';
ids = ones(size(predScores));
ids(referencePredScores>predScores) = 2;
hold on
for i=1:mcCnt
    plot(referencePredScores(i), predScores(i), [cols(ids(i)) '*'])
end

figure
plot(yc, '-k')
hold on
plot(bestOutput, 'r--')
title("Best model output vs noisy reference output")
legend("Noisy reference output", "Model output")

function [L] = ML(A, B, C, Q, R, Phi, Phi_base, Psi1, Psi2, Sigma, Theta, N)
    Gamma1 = [A B];
    L = (N*log(det(R))+sum(diag(R\(Theta-C*Psi2.'-Psi2*C.'+C*Phi*C.')))+N*log(det(Q))+sum(diag(Q\(Phi_base-Gamma1*Psi1.'-Psi1*Gamma1.'+Gamma1*Sigma*Gamma1.'))));
%     L1 = sum(diag(R\(Theta-C*Psi2.'-Psi2*C.'+C*Phi*C.')))
%     L11 = N*log(det(R))+L1
%     L2 = sum(diag(Q\(Phi_base-Gamma1*Psi1.'-Psi1*Gamma1.'+Gamma1*Sigma*Gamma1.')))
%     L21 = N*log(det(Q))+L2
    L = -0.5*L;
end

function L = ML2(A, B, C, Q, R, xtNs, PtNs, MtNs, u, y)
    N = size(y, 2);
    L = N*(log(det(Q))+log(det(R)));
    L1 = 0;
    L2 = 0;
%     test = zeros(size(A));
%     test2 = zeros(size(A));
    for t=1:N
        yt = y(:, t);
        xt = xtNs(:, t);
        L1 = L1 + sum(diag(R\((yt-C*xt)*yt.'-yt*(C*xt).'+C*(xt*xt.'+PtNs(:, :, t))*C.')));
        if t>1
            xtm1 = xtNs(:, t-1);
            utm1 = u(:, t-1);
%             test = test + (xt*xtm1.'+MtNs(:, :, t))*A.'+xt*(B*utm1).';
%             test2 = test2 + xt*(utm1).';
            L2 = L2 + sum(diag(Q\((xt*xt.'+PtNs(:, :, t))-(xt*xtm1.'+MtNs(:, :, t))*A.'-xt*(B*utm1).'-A*(xtm1*xt.'+MtNs(:, :, t).')+A*((xtm1*xtm1.')+PtNs(:, :, t-1))*A.'+A*xtm1*utm1.'*B.'-B*utm1*(xt.'-xtm1.'*A.'-utm1.'*B.'))));
        end
    end
    L = -0.5*(L+L1+L2);
end

function [L] = ML3(Gamma, Pi, Phi, Psi, Sigma, x0N, P0N, mu, Lambda, N)
    L = log(det(Lambda))+sum(diag(Lambda/(x0N*x0N.'+P0N-x0N*mu.'-mu*x0N.'+mu*mu.')));
    L = L + N*log(det(Pi))+sum(diag(Pi\(Phi-Psi*Gamma.'-Gamma*Psi.'+Gamma*Sigma*Gamma.')));
    L = -0.5*L;
end

function [cost] = predictionErrorCost(C, D, xttm1s, y, u)
    cost = 0;
    errs = [];
    errs2 = [];
    for t = 1:size(y, 2)
        tmp = y(:, t)-C*(xttm1s(:, t))-D*u(:, t);
        errs = [errs tmp];
        errs2 = [errs2 tmp.'*tmp];
        cost = cost + tmp.'*tmp;
    end
    cost = cost/numel(y);
end

function [cost] = predictionErrorCost2(A, B, C, D, K, x0, y, u)
    ys = dlsim(A-K*C, [B-K*D K], C, [D zeros(size(D, 1))], [u.' y.'], A*x0);
    tmp = (y-ys.');
    cost = 0;
    for t=1:size(y, 2)
        cost = cost + tmp(:, t).'*tmp(:, t);
    end
    
    cost = cost/numel(y);
end

function [xttm1s] = robustKalmanForward(A, B, C, D, Q, R, S, mu, Lambda, u, y)
    N = size(u, 2);
    nx = size(A, 1);
    ny = size(y, 1);
    rootPtts = zeros([size(A) N]);
    rootPtts(:, :, 1) = zeros(size(A));
    xtts = zeros(size(mu, 1), N);
    xtest = zeros(size(mu, 1), N);
    xttm1s = zeros(size(mu, 1), N);
    %xtts(:, 1) = mu;
    %xttm1s(:, 1) = mu;
    %S = zeros(size(S));
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
    rootPttm1 = R1(1:nx, :).';
    Pttm1 = rootPttm1.'*rootPttm1;
    xttm1 = Abar*mu+SR*C*mu;
    [~, R2] = qr([rootR, C*rootPttm1;zeros(nx, ny), rootPttm1].');
    rootPtts(:, :, 1) = R2(ny+1:end, ny+1:end).';
    Kt = Pttm1*C.'/(C*Pttm1*C.'+R);

    xtts(:, 1) = xttm1+Kt*(y(:, 1)-C*xttm1-D*u(:, 1));
    xttm1s(:, 1) = xttm1;
    xtest(:, 1) = xttm1;
    
    for t=2:N
        [~, R1] = qr([(Abar*rootPtts(:, :, t-1)).';rootQ.']);
        rootPttm1 = R1(1:nx, :).';
        Pttm1 = rootPttm1.'*rootPttm1;
        rootPtp1ts(:, :, t-1) = rootPttm1;
        xttm1 = Abar*xtts(:, t-1)+Bbar*u(:, t-1)+SR*y(:, t-1);
        [~, R2] = qr([rootR, C*rootPttm1;zeros(nx, ny), rootPttm1].');
        rootPtts(:, :, t) = R2(ny+1:end, ny+1:end).';
        Kt = Pttm1*C.'/(C*Pttm1*C.'+R);

        xtts(:, t) = xttm1+Kt*(y(:, t)-C*xttm1-D*u(:, t));
        xttm1s(:, t) = xttm1;
        xtest(:, t) = Abar*xtest(:, t-1)+Bbar*u(:, t-1)+SR*y(:, t-1);
    end
    
%         for t=2:N
%             [~, R1] = qr([(A*rootPtts(:, :, t-1)).';rootQ.']);
%             rootPttm1 = R1(1:nx, :).';
%             Pttm1 = rootPttm1*rootPttm1.';
%             %test1 = A*(rootPtts(:, :, t-1)*rootPtts(:, :, t-1).')*A.'+Q;
%             rootPtp1ts(:, :, t-1) = rootPttm1;
%             xttm1 = A*xtts(:, t-1)+B*u(:, t-1);
%             [~, R2] = qr([rootR C*rootPttm1;zeros(nx, ny) rootPttm1].');
%             rootPtts(:, :, t) = R2(ny+1:end, ny+1:end).';
%             Ptt = rootPtts(:, :, t)*rootPtts(:, :, t).';
%             Kt = Pttm1*C.'/(C*Pttm1*C.'+R);
%             %test2 = Pttm1-Kt*C*Pttm1;
%             
%             xtts(:, t) = xttm1+Kt*(y(:, t)-C*xttm1-D*u(:, t));
%             xttm1s(:, t) = xttm1;
%         end
end

function [Sigma, K] = getSSCov(A, C, Q, R)
    Sigma = Q;
    prevSigma = inf;
    tries = 0;
    while tries < 100
        K = A*Sigma*C.'/(C*Sigma*C.'+R);
        Sigma = A*Sigma*A.'+Q-K*C*Sigma*A.';
        if max(abs(prevSigma-Sigma)) < 1e-10
            break
        end
        %disp(max(abs(prevSigma-Sigma)))
        prevSigma = Sigma;
        tries = tries + 1;
    end
end

function [ys] = sim(C, D, xs, us)
    ys = zeros(size(C, 1), size(us, 2));
    for t=1:size(us, 2)
        ys(:, t) = C*xs(:, t)+D*us(:, t);
    end
end

function [ys] = sim2(A, B, C, D, K, x0, ym, us)
    Ac = A-K*C;
    Bc = B-K*D;
    ys = zeros(size(C, 1), size(us, 2));
    x = A*x0;
    for t=1:size(us, 2)
        y = C*x+D*us(:, t);
        x = Ac*x+Bc*us(:, t)+K*ym(:, t);
        ys(:, t) = y;
    end
end

function [n] = filteredNoise(rows, cols, amp, fpass, fs)
    n = 2*rand(rows, cols)-1;
    if fpass ~= 0
        n = lowpass(n, fpass, fs);
    end
    n = n*amp/max(abs(n), [], 'all');
end

function [X, bins] = CreateMultisine(f_min, f_max, N, fs, phase_type, rms_needed)

Xm = zeros(1, N);
Xp = zeros(1, N);

f_res = fs/N;

bins = (f_min/f_res):(f_max/f_res);
K = length(bins);
for i = bins
    Xm(i+1) = 1;
end

if phase_type == "schroeder"
    for i = bins
        Xp(i+1) = i*(i+1)*pi/K;
    end
elseif phase_type == "random"
    Xp = rand(1, N)*2*pi;
elseif phase_type == "linear"
    for i = bins
        Xp(i+1) = i*pi;
    end
elseif phase_type == "constant"
else
    assert(false, 'Invalid phase_type! Should be "schroeder", "random", "linear" or "constant"');
end

figure
plot((0:N-1)*fs/N, abs(Xm.*exp(1j*Xp)));

X = N*real(ifft(Xm.*exp(1j*Xp)));
X = X*rms_needed/rms(X);

end