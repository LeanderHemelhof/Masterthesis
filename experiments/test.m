clear
close all
clc
rng(48559496)

Ts = 1/50;
Aref = [0.5 0; 9.61e-3 0.8];
Bref = [1; 0];
Cref = [1 961];
Dref = 0;

[V, Aref2] = eig(Aref);
%x'=inv(V)x => Vx'=x =>
%Vx'(k+1)=AVx'(k)+Bu(k)
%    y(k)=CVx'(k)+Du(k)
Bref2 = V\Bref;
Cref2 = Cref*V;
Dref2 = Dref;

%ref = ss(Aref, Bref, Cref, Dref, Ts);
ref = ss(Aref2, Bref2, Cref2, Dref2, Ts);
t = 0:Ts:10;
u = 30*sin(t)+20*sin(3.86*t);%t;%ones(size(t));
un = u+sqrt(3)*randn(size(t));
N = length(t)-1;
y = lsim(ref, un, t);
y2 = y.'+sqrt(0.5)*randn(size(u));
% Aref = [1 1 0;0 1 1;0 0 0];
% Bref = [0;0;1];
% Cref = [1 0 0];
% Dref = 0;
% 
% N = 60;
% ref = ss(Aref, Bref, Cref, Dref, 1);
% [y, t] = impulse(ref, N);
% u = ones(1, N);
% % u = zeros(1, N);
% % u(1) = 1;
% % u(10) = -2;
% % u(20) = 10;
% % u(30) = -9;
% y2 = conv(y, u);
% y2 = y2(1:(N+1))+sqrt(0.5)*randn(N+1, 1);
% y2 = y2.';

figure
plot(t, y2)

A = Aref+0.5*randn(size(Aref));
B = Bref+0.5*randn(size(Bref));
C = Cref+0.5*randn(size(Cref));
D = Dref+0.1*randn(size(Dref));
mu0 = [0];
Q = var(un-u);%3;%randn(size(A)).*eye(size(A, 1));
R = var(y2-y.');%0.5;
%u = [0 u];
best = struct('cost', -inf);
MAX_ITERS = 100;
for oom = -4:4
    foundBetter = false;
    for id = 1:10
        A = randn(size(Aref)).*eye(size(Aref));%(10^oom)*randn(size(Aref));
        B = (10^oom)*randn(size(Bref));
        C = (10^oom)*randn(size(Cref));
        [A, B, C, D, mu0, Qtot, cost] = findModel(un, y2, 3, R, 2, A, B, C, D, false, MAX_ITERS);
        if cost > best.cost
            best.A = A;
            best.B = B;
            best.C = C;
            best.D = D;
            best.mu0 = mu0;
            best.Qtot = Qtot;
            best.cost = cost;
            foundBetter = true;
            fprintf("Found better option: cost=%.3e (oom=%i with id %u)\n", cost, oom, id)
        end
    end
    if ~foundBetter
        fprintf("Found no better option during the last oom (%i). Stopping early.\n", oom)
        %break
    end
end

A = best.A;
B = best.B;
C = best.C;
D = best.D;
mu0 = best.mu0;
Qtot = best.Qtot;

%[A, B, C, D, mu0, Qtot, cost] = findModel(un, y2, 3, R, 1, A, B, C, D, true, MAX_ITERS);
% vOld = -inf;
% for j = 1:MAX_ITERS
%     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
%     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
%     El1_old = 0;%log(abs(det(Sigma0)))+sum(diag(Sigma0\(PNks{1}+(xNks(:, 1)-mu0)*(xNks(:, 1)-mu0).')));
%     if isnan(El1_old)
%         El1_old = 0;
%     end
%     El2_old = N*log(abs(det(Q)))+sum(diag(Q\(Sxx-Sxb*A.'-A*Sxb.'-Sxu1*B.'-B*Sxu1.'+A*Sbu*B.'+B*Sbu.'*A.'+A*Sbb*A.'+B*Suu1*B.')));
%     if isnan(El2_old)
%         El2_old = 0;
%     end
%     El3_old = N*log(abs(det(R)))+sum(diag(R\(Syy-Syx*C.'-C*Syx.'+Syu*D.'-D*Syu.'+C*Sxu2*D.'+D*Sxu2.'*C.'+C*Sxx*C.'+D*Suu2*D.')));
%     if isnan(El3_old)
%         El3_old = 0;
%     end
%     v = -0.5*(El1_old+El2_old+El3_old);
%     if abs((v-vOld)/vOld) < 1e-3% || vOld > v
%         %break
%     end
%     vOld = v;
%     fprintf("(%u) Old log likelihood: %.3f\n", j, v);
%     
%     %% Maximization steps
%     
%     AB = [Sxb Sxu1]/[Sbb Sbu;Sbu.' Suu1];
%     n = size(A, 1);
%     A = AB(:, 1:n);%Sxb/Sbb;
%     B = AB(:, n+1:end);
% %     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
% %     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
% %     AB = [Sxb Sxu1]/[Sbb Sbu;Sbu.' Suu1];
% %     B = AB(:, n+1:end);
% %     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
% %     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
% %     CD = [Syx Syu]/[Sxx Sxu2;Sxu2.' Suu2];
% %     C = CD(:, 1:size(C, 2));%Syx/Sxx;
% %     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
% %     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
% %     CD = [Syx Syu]/[Sxx Sxu2;Sxu2.' Suu2];
% %     D = CD(:, size(C, 2)+1:end);
% %     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
% %     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
% %     mu0 = xNks(:, 1);
% %     [xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, un, y2);
% %     [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, un, y2);
% %     Q=(Sxx-Sxb*A.'-A*Sxb.'-Sxu1*B.'-B*Sxu1.'+A*Sbu*B.'+B*Sbu.'*A.'+A*Sbb*A.'+B*Suu1*B.')/N;
%     %Q=(Sxx-Sxb*A.'-A*Sxb.'+A*Sbb*A.')/N;
%     %[xNks, PNks, PNkkm1s] = KalmanSmooth(A, B, C, D, mu0, Q, R, u, y2);
%     %[Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, u, y2);
%     %R=(Syy-Syx*C.'-C*Syx.'+Syu*D.'-D*Syu.'+C*Sxu2*D.'+D*Sxu2.'*C.'+C*Sxx*C.'+D*Suu2*D.')/N;
%     %R=(Syy-Syx*C.'-C*Syx.'+C*Sxx*C.')/N;
% %     A = Anew;
% %     %B = Bnew;
% %     C = Cnew;
% %     %D = Dnew;
% %     mu0 = mu0new;
% %     Q = Qnew;
%     %El1 = log(det(Sigma0))+sum(diag(Sigma0\(PNks{1}+(xNks(:, 1)-mu0)*(xNks(:, 1)-mu0).')));
%     %El2 = 0;%N*log(det(Q))+sum(diag(Q\(Sxx-Sxb*A.'-A*Sxb.'-Sxu1*B.'-B*Sxu1.'+A*Sbu*B.'+B*Sbu.'*A.'+A*Sbb*A.'+B*Suu1*B.')));
%     %El3 = N*log(det(R))+sum(diag(R\(Syy-Syx*C.'-C*Syx.'+Syu*D.'-D*Syu.'+C*Sxu2*D.'+D*Sxu2.'*C.'+C*Sxx*C.'+D*Suu2*D.')));
% end

model = ss(A, B, C, D, Ts);
disp(model)
un = u.*cos(8.697*t)+sqrt(3)*randn(size(t));
N = length(t)-1;
yref = lsim(ref, un, t);
y = lsim(model, un, t, mu0);
figure
plot(t, yref, 'x')
hold on
plot(t, y, '--')
