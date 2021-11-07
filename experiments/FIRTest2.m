clear
close all
clc

fs = 50;
N = 20000;
numIn = 2; numOut = 2;
Aref = [0.3 0.12 0.4; 0.7 0.284 0.1; 0.6 0.4 0.05]; Bref = [7 1;18 5;29 7]; Cref = [25 6 9; 13 19.6 17.5]; Dref=[7.16 13;1.2 28];
model = ss(Aref, Bref, Cref, Dref, 1/fs);
[ubase, bins] = CreateMultisine(0, fs/2, N, fs, 'schroeder', 0.2);
ytot = lsim(model, repmat(ubase, numIn, 10));
ys = splitOutput(ytot.', 10);
ytot = ys(:, :, end);
pow = [rms(ytot(1, :))^2; rms(ytot(2, :))^2];

maxVar = 100;
list = zeros(2, maxVar);
list2 = zeros(2, maxVar);
list3 = zeros(1, maxVar);
sigma2s = linspace(500, 5000, maxVar);
x = zeros(2, maxVar);
for id=1:maxVar
    sigma2 = sigma2s(id);
    sqErrs = estimateWithNoisePower(model, 2, 2, ubase, sigma2, 20, 0.1*sigma2);
    relSqErrs = sqErrs./ytot;
    list(:, id) = max(sqErrs, [], 2);
    list2(:, id) = max(relSqErrs, [], 2);
    x(1, id) = pow(1)/sigma2;
    x(2, id) = pow(2)/sigma2;
%     figure
%     for row=1:numOut
%         subplot(numOut, 1, row)
%         plot(sqErrs(row, :))
%     end
end
figure
subplot(2, 1, 1)
plot(x(1, :), list(1, :))
subplot(2, 1, 2)
plot(x(2, :), list(2, :))
figure
subplot(2, 1, 1)
plot(x(1, :), list2(1, :))
subplot(2, 1, 2)
plot(x(2, :), list2(2, :))
%estimatePeriodsNeeded(model, u, 0.5, 20000);


function [sqErrs] = estimateWithNoisePower(ref, numIn, numOut, ubase, sigma2, maxReps, maxMSE)
    
    m = numIn;
    p = numOut;
    N = size(ubase, 2);
    reps = estimatePeriodsNeeded(ref, repmat(ubase, numIn, 1), sigma2, maxMSE);
    reps = min(reps, maxReps);
    
    Gk = zeros(p, m, N);
%     figure
    for inputId=1:m
        u = zeros(m, N);
        u(inputId, :) = ubase;
        y = lsim(ref, repmat(u, 1, reps+100));
        ys = splitOutput(y.', reps+100);
        y = ys(:, :, reps:end);
        y = y + sqrt(sigma2)*randn(size(y));
%         figure
        for row=1:p
%             subplot(p, 1, row)
%             title(['Output ' num2str(row)])
%             hold on
%             for i=1:reps
%                 plot(ys(row, :, i))
%             end
            ffts = zeros(size(y, 3), size(ubase, 2));
            for run=1:size(y, 3)
                ffts(run, :) = fft(y(row, :, run));
            end
            Gk(row, inputId, :) = mean(ffts, 1)./fft(ubase);
%             subplot(p, m, inputId+(row-1)*m)
%             plot(db(reshape(Gk(row, inputId, :), 1, [])))
        end
    end
    
    ytot = lsim(ref, repmat(ubase, m, reps));
    %y = y+sqrt(sigma2)*randn(size(y));
    ys = splitOutput(ytot.', reps);
    ytot = ys(:, :, end);
    [A, B, C, D] = estimateModel(Gk, 3, 2.5e-3, 100);
    model = ss(A, B, C, D, 1/50);
    ym = lsim(model, repmat(ubase, m, reps));
    ysm = splitOutput(ym.', reps);
    ym = ysm(:, :, end);
    sqErrs = (ytot-ym).^2;
end

function [reps] = estimatePeriodsNeeded(model, u, sigma2, maxVar)
    reps = 1;
    vari = inf;
    while vari > maxVar && reps<200
        reps = reps+1;
        y = lsim(model, repmat(u, 1, reps));
        y = y+sqrt(sigma2)*randn(size(y));
        ys = splitOutput(y.', reps);
        error = ys(:, :, end)-ys(:, :, end-1);
        meanSqError = sum(error.^2, 'all')/numel(error);
        vari = meanSqError - 2*sigma2;
        %fprintf("Reps=%u: mse=%.3e\n", reps, vari)
    end
    reps = reps-1;
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