clear
close all
clc

%Aref = [0.3 0.12 0.4; 0.7 0.284 0.1; 0.6 0.4 0.05]; Bref = [7;18;29]; Cref = [25 6 9; 13 19.6 17.5]; Dref=[7.16;1.2];
Aref = [0.3 0.12 0.4; 0.7 0.284 0.1; 0.6 0.4 0.05]; Bref = [7 1;18 5;29 7]; Cref = [25 6 9; 13 19.6 17.5]; Dref=[7.16 13;1.2 28];
eigA = eig(Aref);
fs = 50;
ref = ss(Aref, Bref, Cref, Dref, 1/fs);
N = 100;
fmin = 0;
fmax = fs/2;%4999;
[ubase, bins] = CreateMultisine(fmin, fmax, N, fs, 'schroeder', 0.2);
p=2;
m=2;
Gk = zeros(p, m, N);
for inputId=1:m
    % t = 0:0.1:5;
    % t(end) = [];
    % u = sin(4*pi*t);
    % N = 50;
    % freqs = ((-N/2):(N/2)-1)*fs/N;
    % fmask = and(abs(freqs) <= fmax, abs(freqs) >= fmin);
    % figure
    % stem(freqs, abs(fftshift(fft(u))))
    u = zeros(m, N);
    u(inputId, :) = ubase;
    [y, t] = lsim(ref, repmat(u, 1, 5));
    t = t(1:N);
    y = y.';
    p = size(y, 1);
    m = size(u, 1);
    %y = reshape(y, [], 5).';
    ys = splitOutput(y, 5);
    y = ys(:, :, end);
    figure
    for row=1:p
        subplot(p, 1, row)
        title(['Output ' num2str(row)])
        hold on
        for i=1:5
            plot(ys(row, :, i))
        end
        Gk(row, inputId, :) = fft(y(row, :))./fft(ubase);
    end
end
figure
plot(t, y);
G2 = fft(y)./fft(u);
% Gk = zeros(p, m, size(G2, 2)/m);
% for k=1:size(G2, 2)/m
%     Gk(:, :, k) = G2(:, m*(k-1)+1:m*k);
% end
M = length(y)/2;
wks = (0:M)*pi/M;
wksx = 0:(pi/M):2*pi;
wksx(end) = [];
amt = 500;
gks = zeros(p, m, amt+1);
gks(:, :, 1) = Dref;
for k=1:amt
    gks(:, :, k+1) = Cref*Aref^(k-1)*Bref;
end
ytest = resp(gks, u, t, fs);
figure
hold on
plot(t, ys(1, :, 1));
plot(t, ytest(1, :));
figure
hold on
plot(t, ys(2, :, 1));
plot(t, ytest(2, :));
Gtest = zeros(p, m, M+1);
Gtest2 = zeros(p, m, M+1);
for k=0:amt
    for wkId=1:length(wks)
        Gtest(:, :, wkId) = Gtest(:, :, wkId) + gks(:, :, k+1)*exp(-1j*k*wks(wkId));
    end
end
% % emjk = exp(-1j*(0:30));
% % Gtest = sum(gks.*emjk)*exp(wks);
for wkId=1:length(wks)
    Gtest2(:, :, wkId) = Cref*(exp(1j*wks(wkId))*eye(size(Aref, 1))-Aref)^-1*Bref+Dref;
end
% G = Cref*(exp(1j*wks)-Aref).^-1*Bref+Dref;
% %G2 = fftshift(G);
% %G2 = G2(fmask);
% % figure
% % stem(freqs, db(abs(G)))
% 
% Gex = zeros(1, 2*M);
% Gex(1:M+1) = G;
% for k = 1:M-1
%     Gex(M+1+k) = conj(G(M+1-k));
% end
% figure
% plot(wksx, db(abs(Gex)))

figure
for row=1:p
    for col=1:m
        subplot(p, m, m*(row-1)+col)
        title(['TF output ' num2str(row) ' and input ' num2str(col)])
        hold on
        plot(wksx, db(abs(reshape(Gk(row, col, :), 1, []))))
        plot(wks, db(abs(reshape(Gtest(row, col, :), 1, []))))
    end
end
q = 50;
r = 50;
his = zeros(p, m, q+r-1);
is = 1:q+r-1;
for k=0:(2*M-1)
    col = 1;
    for i=is
        his(:, :, i) = his(:, :, i) + Gk(:, :, k+1)*exp(1j*2*pi*i*k/(2*M));
    end
end
his = his/(2*M);
H = zeros(q*p, r*m);
rid = 1; cid = 1;
for row = 1:q
    H(rid+(0:p-1), :) = reshape(real(his(:, :, row+(0:r-1))), p, []);
    rid = rid+p;
end
[U, S, V] = svd(H);
svs = diag(S);
n = 1;
rel = svs(n+1)/svs(1);
while rel > 1e-4
    n = n+1;
    rel = svs(n+1)/svs(1);
end
Ss = S(1:n, 1:n);
Us = U(:, 1:n);
J1 = [eye((q-1)*p) zeros((q-1)*p, p)];
J2 = [zeros((q-1)*p, p) eye((q-1)*p)];
J3 = [eye(p) zeros(p, (q-1)*p)];
A  = real(pinv(J1*Us)*J2*Us);
C  = J3*Us;
T = pinv(Cref)*C;
zs = exp(1j*wks);
chi = zeros(p*(M+1), n+p);
for k = 0:M
    chi(k*p+(1:p), :) = [C*(zs(k+1)*eye(n)-A)^-1 eye(p)];
end
Ghat = zeros(m, p*size(Gk, 2));
for k = 0:M
    Ghat(:, k*p+(1:p)) = Gk(:, :, k+1).';
end
Ghat = Ghat.';
BD = pinv([real(chi);imag(chi)])*[real(Ghat);imag(Ghat)];
B = BD(1:n, :);
T2 = Bref*pinv(B);
D = BD(n+1:end, :);
model = ss(A, B, C, D, 1/fs);
ymod = lsim(model, repmat(ubase, m, 5));
ys2 = splitOutput(ymod.', 5);
ymod = ys2(:, :, end);

ytot = lsim(ref, repmat(ubase, m, 5));
ystot = splitOutput(ytot.', 5);
ytot = ystot(:, :, end);
figure
for row=1:p
    subplot(p, 1, row)
    title(['Output ' num2str(row)])
    hold on
    for i=1:5
        plot(t, (ymod(1, :)-ytot(1, :)).^2.');
    end
end


function [ys] = splitOutput(y, amt)
    lenPer = size(y, 2)/amt;
    ys = zeros(size(y, 1), lenPer, amt);
    for k=1:amt
        ys(:, :, k) = y(:, ((1+(k-1)*lenPer):(k*lenPer)));
    end
end

function [y] = resp(gks, u, t, fs)
    y = zeros(size(gks, 1), length(t));
    for time=round(t.'*fs)
        for k=0:length(gks)-1
            if time-k < 0
                break
            end
            y(:, time+1) = y(:, time+1) + gks(:, :, k+1)*u(:, time-k+1);
        end
    end
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

if bins(1) == 0
    Xm(1) = Xm(1)/2;
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

X = N*real(ifft(Xm.*exp(1j*Xp)));
X = X*rms_needed/rms(X);

end