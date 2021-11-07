%x(t+1)=a*x(t)+b*x(t)/(1 +x(t)^2)+c*cos(1.2t) +v(t),
%y(t)=d*x^2(t)+e(t)    v and e zero mean normally distributed with variance
%q and r respectively
clear
close all
clc
rng(7654398)

MC_amt = 100;
particle_amt = 100;
M = particle_amt;
MAX_ITERS = 1000;
aref = 0.5; x0ref = 16; qref = 0.005; rref = 0.01;
params_ref = [aref];
N = 10;
y = zeros(1, N);
x = x0ref;
for t = 1:N
    x = f(x, aref, t-1)+sqrt(qref)*randn(1);
    y(t) = g(x, aref, t)+sqrt(rref)*randn(1);
end
scores = zeros(MC_amt, 1);
endParams = zeros(MC_amt, length(params_ref));
%% TESTING
% params = params_ref;
% params(6) = 0.0005;
params = ((rand(size(params_ref))-0.5).*params_ref)+params_ref;
%params(3) = 0.0005*rand(1)+0.001;
[wit, xTildes] = particleFilter(N, M, y, params);
swit = particleSmoother(N, M, wit, xTildes, params);
wijt = zeros(M, M, N-1);
[a] = toVals(params);
q = 0.005; r = 0.01; x0 = 16;

for t = 1:(N-1)
    lmask = wit(:, t) > 1e-10;
    v1 = xTildes(:, :, t+1).'-f(xTildes(:, lmask, t), a, t);
    offset2 = min(v1.^2, [], 2)/(2*q);
    denoms = sum(repmat(wit(lmask, t).', size(v1, 1), 1).*normpdf_offset(v1, 0, sqrt(q), offset2), 2);
    is = 1:M;
    is = is(lmask);
    for i = is
        p = normpdf_offset(xTildes(:, :, t+1).'-f(xTildes(:, i, t), a, t), 0, sqrt(q), offset2);
        wijt(i, :, t) = (wit(i, t)*swit(:, t+1).*p./denoms).';
    end
end

params2 = params_ref;
as = params2(1)*linspace(0.5, 1.5, 101);
Qs = [];
for a=as
    Qs = [Qs Q(a, wit, wijt, swit, xTildes, y)];
end
figure
plot(as, Qs);

% % params = params_ref;
% % params(6) = 0.0005;
% bs = params(2)*linspace(0.5, 1.5, 101);
% as = params(1)*linspace(0.5, 1.5, 101);
% Qs = [];
% params2 = params;
% for bId = 1:length(bs)
%     params2(2) = bs(bId);
%     for aId = 1:length(as)
%         params2(1) = as(aId);
%         Qs(aId, bId) = Q(params2, wit, wijt, swit, xTildes, y);
%     end
% end
% figure
% surf(as, bs, Qs)
%% Gradient descent
        alpha = 1;
        gdIt = 1;
        current = Q(params, wit, wijt, swit, xTildes, y);
        while true%abs((prev-current)/current) > 1e-3 && gdIt < MAX_ITERS
            [a, x0, q, r] = toVals(params);
            dI1 = zeros(size(params));
            ddI1 = zeros(length(params), length(params));
            for i = 1:M
                %I1 = sum(swit(:, 1).'.*log(normpdf(xTildes(:, :, 1)-f(x0, a, b, c, d, 0), 0, sqrt(q))));
                x = xTildes(:, i, 1)-f(x0, a, 0);
%                 dI1 = dI1 + swit(i, 1)*(x)/q*df(x0, a, b, c, d, 0, true);
%                 dI1(6) = dI1(6) + swit(i, 1)*(1/(pi)+(x)^2)/(2*q^2);
                dif = -df(x0, a, 0, true);
                dI1 = dI1 + swit(i, 1)*(-x*dif/q);
                dI1(6) = dI1(6) + swit(i, 1)*((x.^2)/q-1)/(2*q);
                
                difdif = -ddf(x0, a, 0, true);
                dSigma2 = zeros(1, length(params));
                %dSigma2(6) = 1;
                for p1 = 1:length(params)%theta_i
                    for p2 = 1:length(params)%theta_j
                        ddI1(p1, p2) = ddI1(p1, p2) - swit(i, 1)*dSigma2(p2)/q^2;%A
                        n1 = ((2*((dif(p2)*q+x*dSigma2(p2))*dif(p1)+x*q*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1)))*q^2;
                        n2 = 2*q*dSigma2(p2)*(2*x*dif(p1)-x^2*dSigma2(p1));
                        ddI1(p1, p2) = ddI1(p1, p2) + swit(i, 1)*(n1-n2)/q^4;%B
                    end
                end
            end
            ddI1 = ddI1*(-0.5);
            dI3 = 0;
            ddI3 = zeros(size(ddI1));
            for t = 1:N
                for i=1:M
                    %I3 = I3 + sum(swit(:, t).'.*log(normpdf(y(t)-g(xTildes(:, :, t), a, b, c, d, t), 0, sqrt(r))));
%                     dI3 = dI3 + swit(i, t)*(y(t)-g(xTildes(:, i, t), a, b, c, d, t))/r*dg(xTildes(:, i, t), a, b, c, d, t, false);
%                     dI3(7) = dI3(7) + swit(i, 1)*(1/(pi)+(y(t)-g(xTildes(:, i, t), a, b, c, d, t))^2)/(2*r^2);
                    x = y(t)-g(xTildes(:, i, t), a, t);
                    dif = -dg(xTildes(:, i, t), a, t, false);
                    dI3 = dI3 + swit(i, t)*(-x*dif/r);
                    dI3(7) = dI3(7) + swit(i, t)*((x.^2)/r-1)/(2*r);
                    
                    difdif = -ddg(xTildes(:, i, t), a, t, false);
                    dSigma2 = zeros(1, length(params));
                    %dSigma2(7) = 1;
                    for p1 = 1:length(params)%theta_i
                        for p2 = 1:length(params)%theta_j
                            ddI3(p1, p2) = ddI3(p1, p2) - swit(i, t)*dSigma2(p2)/r^2;%A
                            n1 = ((2*((dif(p2)*r+x*dSigma2(p2))*dif(p1)+x*r*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1)))*r^2;
                            n2 = 2*r*dSigma2(p2)*(2*x*dif(p1)-x^2*dSigma2(p1));
                            ddI3(p1, p2) = ddI3(p1, p2) + swit(i, t)*(n1-n2)/r^4;%B
                        end
                    end
                end
            end
            dI2 = 0;
            ddI3 = ddI3*(-0.5);
            ddI2 = zeros(size(ddI1));
            for t = 1:(N-1)
                for i = 1:M
                    %for j = 1:M
%                         offset = min(((xTildes(:, j, t+1)-f(xTildes(:, :, t), a, b, c, d, t)).^2)/(2*q));
%                         denom = sum(wit(:, t).'.*normpdf_offset(xTildes(:, j, t+1)-f(xTildes(:, :, t), a, b, c, d, t), 0, sqrt(q), offset));
%                         if denom == 0
%                             continue
%                         end
%                         p = normpdf_offset(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q), offset);
%                         wij = wit(i, t)*swit(j, t+1)*p/denom;
%                         wijt(i, j, t) = wij;
%                         I2 = I2 + wij*logp(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q));
                        %dI2 = dI2 + wijt(i, j, t)*(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t))/q*df(xTildes(:, i, t), a, b, c, d, t, false);
                        jmask = wijt(i, :, t) ~= 0;
                        if sum(jmask) == 0
                            continue
                        end
                        x = xTildes(:, jmask, t+1)-f(xTildes(:, i, t), a, t);
                        dif = -df(xTildes(:, i, t), a, t, false);
                        dI2 = dI2 - sum(wijt(i, jmask, t).*(x/q), 2)*dif;
                        dI2(6) = dI2(6) + sum(wijt(i, jmask, t).*((x.^2)/q-1)/(2*q), 2);
                        
                        difdif = -ddf(xTildes(:, i, t), a, t, true);
                        dSigma2 = zeros(1, length(params));
                        %dSigma2(6) = 1;
                        sot = sum(wijt(i, jmask, t))/q^2;
                        for p1 = 1:length(params)%theta_i
                            for p2 = 1:length(params)%theta_j
                                n1 = (2*((dif(p2)*q+x*dSigma2(p2))*dif(p1)+x*q*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1))*q^2;
                                n2 = 2*q*dSigma2(p2)*(2*x*dif(p1)-x.^2*dSigma2(p1));
                                if sum(n1 ~= n2) == 0 && dSigma2(p2) == 0
                                    continue
                                end
                                ddI2(p1, p2) = ddI2(p1, p2) + ((wijt(i, jmask, t)*(n1-n2).')/q^4)-(sot*dSigma2(p2));%B+A
                            end
                        end
                    %end
                end
            end
            if gdIt > 1
                prevGrad = grad;
            end
            grad = dI1 + dI2 + dI3;
            H = ddI1+ddI2+ddI3;
            Hi = inv(H);
            if gdIt == 1
                d_n = grad;
            else
                beta_n=-(grad*(grad-prevGrad).')/(prevGrad*prevGrad.');
                beta_n = max(beta_n, 0);
                d_n = grad + beta_n*d_n;
            end
            delta_n = (grad*grad.'/abs(d_n*H*d_n.'));
            p = delta_n*d_n;
            p(3:end) = 0;
            newParams = params+alpha*p;
%             p = (Hi*grad.').';
%             newParams = params+alpha*p;%*grad;
            
            newParams(6) = max(newParams(6), 1e-6);
            Qnew = Q(newParams, wit, wijt, swit, xTildes, y);
            tries = 0;
            while Qnew < current && tries < 20
                alpha = 0.1*alpha;
                newParams = params+alpha*p;
                newParams(6) = max(newParams(6), 1e-6);
                Qnew = Q(newParams, wit, wijt, swit, xTildes, y);
                tries = tries + 1;
            end
            alpha = 4*alpha;
            prev = current;
            current = Qnew;
            params = newParams;
            fprintf("\t(%u) New cost: %.3e (relative change of %.3e and alpha=%.3e/%.3e)\n", gdIt, Qnew, abs((prev-current)/current), alpha, alpha*p*grad.');
            gdIt = gdIt + 1;
        end
%% END TESTING

for id = 1:MC_amt
    %params = params_ref;
    params = ((rand(size(params_ref))-0.5).*params_ref)+params_ref;
    params(6) = 0.0005*rand(1)+0.001;
    %params(7) = rref;
    alpha = 1;
    for it = 1:MAX_ITERS
        a = params(1);
        x0 = params(2);
        q = params(3);
        r = params(4);
        %% particle filter
        [wit, xTildes] = particleFilter(N, M, y, params);
%         xit = zeros(1, M, N);
%         wit = zeros(M, N);
%         %wit2 = zeros(M, N);
%         % start
%         xTilde = zeros(1, M);
%         xTildes = zeros(1, M, N);
%         for t=1:N
%             % generate new particles
%             if t~=1
%                 xTilde = f(xit(:, :, t-1), a, b, c, d, t-1)+sqrt(q)*randn(size(xTilde));
%             else
%                 % Initialize particles x^i_0
%                 xi0 = x0+sqrt(q)*randn(size(xTilde));
%                 xTilde = f(xi0, a, b, c, d, 0)+sqrt(q)*randn(1);
%             end
%             offset = min(((y(t)-g(xTilde, a, b, c, d, t)).^2))/(2*r);
%             wit(:, t) = normpdf_offset(y(t)-g(xTilde, a, b, c, d, t), 0, sqrt(r), offset);
%             soe = sum(wit(:, t));
%             wit(:, t) = wit(:, t)/soe;
% %             wit(:, t) = normpdf(y(t)-g(xTilde, a, b, c, d, t), 0, sqrt(r));
% %             offset = min(((y(t)-g(xTilde, a, b, c, d, t)).^2))/(2*r);
% %             wit2(:, t) = normpdf_offset(y(t)-g(xTilde, a, b, c, d, t), 0, sqrt(r), offset);
% %             soe = sum(wit(:, t));
% %             soe2 = sum(wit2(:, t));
% %             wit2(:, t) = wit2(:, t)/soe2;
% %             if soe == 0  % estimate too bad, all probabilities are too low
% %                 % Keep tilde values like a uniform sampling was done
% %                 xit(:, :, t) = xTilde;
% %             else
% %                 wit(:, t) = wit(:, t)/soe;
% %                 xit(:, :, t) = randsample(xTilde, M, true, wit(:, t));
% %             end
%             xit(:, :, t) = randsample(xTilde, M, true, wit(:, t));
%             xTildes(:, :, t) = xTilde;
%         end
        
        %% particle smoother
        [swit] = particleSmoother(N, M, wit, xTildes, params);
%         swit = zeros(M, N);
%         swit(:, N) = wit(:, N);
%         for t=(N-1):-1:1
%             indices = wit(:, t) > 1e-10;%wit(:, t) ~= 0;
%             if sum(indices) == 0
%                 continue
%             end
%             ids = 1:M;
%             kmask = ids(swit(:, t+1) > 1e-10);%swit(:, t+1) ~= 0);
%             xmat = xTildes(:, kmask, t+1).'-f(xTildes(:, indices, t), a, b, c, d, t);
%             offset = min(((xmat).^2), [], 2)/(2*q);
%             vk = sum(repmat(wit(indices, t).', size(xmat, 1), 1).*normpdf_offset(xmat, 0, sqrt(q), offset), 2);
%             for i = ids(indices)
%                 value = swit(kmask, t+1).'*(normpdf_offset((xTildes(:, kmask, t+1)-f(xTildes(:, i, t), a, b, c, d, t)).', 0, sqrt(q), offset)./vk);
% %                 value = 0;
% %                 for k = ids(swit(:, t+1) ~= 0)
% %                     offset = min(((xTildes(:, k, t+1)-f(xTildes(:, indices, t), a, b, c, d, t)).^2)/(2*q));
% %                     vk = sum(wit(indices, t).'.*normpdf_offset(xTildes(:, k, t+1)-f(xTildes(:, indices, t), a, b, c, d, t), 0, sqrt(q), offset));
% %                     value = value + swit(k, t+1)*normpdf_offset(xTildes(:, k, t+1)-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q), offset)/vk;
% %                 end
%                 swit(i, t) = wit(i, t)*value;
%             end
%         end
        
        wijt = zeros(M, M, N-1);
        %wijt2 = zeros(M, M, N-1);
        for t = 1:(N-1)
            lmask = wit(:, t) > 1e-10;
            v1 = xTildes(:, :, t+1).'-f(xTildes(:, lmask, t), a, b, c, d, t);
            offset2 = min(v1.^2, [], 2)/(2*q);
            denoms = sum(repmat(wit(lmask, t).', size(v1, 1), 1).*normpdf_offset(v1, 0, sqrt(q), offset2), 2);
            is = 1:M;
            is = is(lmask);
            for i = is
                p = normpdf_offset( xTildes(:, :, t+1).'-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q), offset2);
                wijt(i, :, t) = (wit(i, t)*swit(:, t+1).*p./denoms).';
%                 for j = 1:M
%                     v1 = xTildes(:, j, t+1)-f(xTildes(:, :, t), a, b, c, d, t);
%                     offset = min(((v1).^2))/(2*q);
%                     denom = sum(wit(:, t).'.*normpdf_offset(v1, 0, sqrt(q), offset));
%                     if denom == 0
%                         continue
%                     end
%                     v2 = xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t);
%                     p = normpdf_offset(v2, 0, sqrt(q), offset);
%                     wij = wit(i, t)*swit(j, t+1)*p/denom;
%                     wijt(i, j, t) = wij;
%                 end
            end
        end
        
        %% Calculate QM
        QM = Q(params, wit, wijt, swit, xTildes, y);
        prev = -inf;
        current = QM;
        %% Gradient descent
        gdIt = 1;
        while true%abs((prev-current)/current) > 1e-3 && gdIt < MAX_ITERS
            [a, b, c, d, x0, q, r] = toVals(params);
            dI1 = zeros(size(params));
            ddI1 = zeros(length(params), length(params));
            for i = 1:M
                %I1 = sum(swit(:, 1).'.*log(normpdf(xTildes(:, :, 1)-f(x0, a, b, c, d, 0), 0, sqrt(q))));
                x = xTildes(:, i, 1)-f(x0, a, b, c, d, 0);
%                 dI1 = dI1 + swit(i, 1)*(x)/q*df(x0, a, b, c, d, 0, true);
%                 dI1(6) = dI1(6) + swit(i, 1)*(1/(pi)+(x)^2)/(2*q^2);
                dif = -df(x0, a, b, c, d, 0, true);
                dI1 = dI1 + swit(i, 1)*(-x*dif/q);
                dI1(6) = dI1(6) + swit(i, 1)*((x.^2)/q-1)/(2*q);
                
                difdif = -ddf(x0, a, b, c, d, 0, true);
                dSigma2 = zeros(1, length(params));
                dSigma2(6) = 1;
                for p1 = 1:length(params)%theta_i
                    for p2 = 1:length(params)%theta_j
                        ddI1(p1, p2) = ddI1(p1, p2) - swit(i, 1)*dSigma2(p2)/q^2;%A
                        n1 = ((2*((dif(p2)*q+x*dSigma2(p2))*dif(p1)+x*q*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1)))*q^2;
                        n2 = 2*q*dSigma2(p2)*(2*x*dif(p1)-x^2*dSigma2(p1));
                        ddI1(p1, p2) = ddI1(p1, p2) + swit(i, 1)*(n1-n2)/q^4;%B
                    end
                end
            end
            ddI1 = ddI1*(-0.5);
            dI3 = 0;
            ddI3 = zeros(size(ddI1));
            for t = 1:N
                for i=1:M
                    %I3 = I3 + sum(swit(:, t).'.*log(normpdf(y(t)-g(xTildes(:, :, t), a, b, c, d, t), 0, sqrt(r))));
%                     dI3 = dI3 + swit(i, t)*(y(t)-g(xTildes(:, i, t), a, b, c, d, t))/r*dg(xTildes(:, i, t), a, b, c, d, t, false);
%                     dI3(7) = dI3(7) + swit(i, 1)*(1/(pi)+(y(t)-g(xTildes(:, i, t), a, b, c, d, t))^2)/(2*r^2);
                    x = y(t)-g(xTildes(:, i, t), a, b, c, d, t);
                    dif = -dg(xTildes(:, i, t), a, b, c, d, t, false);
                    dI3 = dI3 + swit(i, t)*(-x*dif/r);
                    dI3(7) = dI3(7) + swit(i, t)*((x.^2)/r-1)/(2*r);
                    
                    difdif = -ddg(xTildes(:, i, t), a, b, c, d, t, false);
                    dSigma2 = zeros(1, length(params));
                    dSigma2(7) = 1;
                    for p1 = 1:length(params)%theta_i
                        for p2 = 1:length(params)%theta_j
                            ddI3(p1, p2) = ddI3(p1, p2) - swit(i, t)*dSigma2(p2)/r^2;%A
                            n1 = ((2*((dif(p2)*r+x*dSigma2(p2))*dif(p1)+x*r*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1)))*r^2;
                            n2 = 2*r*dSigma2(p2)*(2*x*dif(p1)-x^2*dSigma2(p1));
                            ddI3(p1, p2) = ddI3(p1, p2) + swit(i, t)*(n1-n2)/r^4;%B
                        end
                    end
                end
            end
            dI2 = 0;
            ddI3 = ddI3*(-0.5);
            ddI2 = zeros(size(ddI1));
            for t = 1:(N-1)
                for i = 1:M
                    %for j = 1:M
%                         offset = min(((xTildes(:, j, t+1)-f(xTildes(:, :, t), a, b, c, d, t)).^2)/(2*q));
%                         denom = sum(wit(:, t).'.*normpdf_offset(xTildes(:, j, t+1)-f(xTildes(:, :, t), a, b, c, d, t), 0, sqrt(q), offset));
%                         if denom == 0
%                             continue
%                         end
%                         p = normpdf_offset(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q), offset);
%                         wij = wit(i, t)*swit(j, t+1)*p/denom;
%                         wijt(i, j, t) = wij;
%                         I2 = I2 + wij*logp(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t), 0, sqrt(q));
                        %dI2 = dI2 + wijt(i, j, t)*(xTildes(:, j, t+1)-f(xTildes(:, i, t), a, b, c, d, t))/q*df(xTildes(:, i, t), a, b, c, d, t, false);
                        jmask = wijt(i, :, t) ~= 0;
                        if sum(jmask) == 0
                            continue
                        end
                        x = xTildes(:, jmask, t+1)-f(xTildes(:, i, t), a, b, c, d, t);
                        dif = -df(xTildes(:, i, t), a, b, c, d, t, false);
                        dI2 = dI2 - sum(wijt(i, jmask, t).*(x/q), 2)*dif;
                        dI2(6) = dI2(6) + sum(wijt(i, jmask, t).*((x.^2)/q-1)/(2*q), 2);
                        
                        difdif = -ddf(xTildes(:, i, t), a, b, c, d, t, true);
                        dSigma2 = zeros(1, length(params));
                        dSigma2(6) = 1;
                        sot = sum(wijt(i, jmask, t))/q^2;
                        for p1 = 1:length(params)%theta_i
                            for p2 = 1:length(params)%theta_j
                                n1 = (2*((dif(p2)*q+x*dSigma2(p2))*dif(p1)+x*q*difdif(p1, p2)))-(2*x*dif(p2)*dSigma2(p1))*q^2;
                                n2 = 2*q*dSigma2(p2)*(2*x*dif(p1)-x.^2*dSigma2(p1));
                                if sum(n1 ~= n2) == 0 && dSigma2(p2) == 0
                                    continue
                                end
                                ddI2(p1, p2) = ddI2(p1, p2) + ((wijt(i, jmask, t)*(n1-n2).')/q^4)-(sot*dSigma2(p2));%B+A
                            end
                        end
                    %end
                end
            end
            if gdIt > 1
                prevGrad = grad;
            end
            grad = dI1 + dI2 + dI3;
            H = ddI1+ddI2+ddI3;
            Hi = inv(H);
            if gdIt == 1
                d_n = grad;
            else
                beta_n=-(grad*(grad-prevGrad).')/(prevGrad*prevGrad.');
                beta_n = max(beta_n, 0);
                d_n = grad + beta_n*d_n;
            end
            delta_n = (grad*grad.'/abs(d_n*H*d_n.'));
            p = delta_n*d_n;
            newParams = params+alpha*p;
%             p = (Hi*grad.').';
%             newParams = params+alpha*p;%*grad;
            
            newParams(3) = max(newParams(3), 1e-6);
            Qnew = Q(newParams, wit, wijt, swit, xTildes, y);
            tries = 0;
            while Qnew < current && tries < 20
                alpha = 0.1*alpha;
                newParams = params+alpha*p;
                newParams(6) = max(newParams(6), 1e-6);
                Qnew = Q(newParams, wit, wijt, swit, xTildes, y);
                tries = tries + 1;
            end
            alpha = 4*alpha;
            prev = current;
            current = Qnew;
            params = newParams;
            fprintf("\t(%u:%u:%u) New cost: %.3e (relative change of %.3e and alpha=%.3e/%.3e)\n", id, it, gdIt, Qnew, abs((prev-current)/current), alpha, alpha*p*grad.');
            gdIt = gdIt + 1;
            if abs(alpha*p*grad.') < 5e-1
                break;
            end
        end
        fprintf("Comparing %.3e (new) and %.3e (old) (difference is %.3e)\n", current, QM, current-QM)
        fprintf("Params: a=%.3f x0=%.3f q=%.3f r=%.3f\n", params)
        if current-QM < 5e-1
            break
        end
        %alpha = 1;%alpha * 1000;
    end
    scores(id) = current;
    endParams(id, :) = params;
end

vars = zeros(size(params));
means = zeros(size(params));
names = ["a";"x0";"q";"r"];
fprintf("Results:\n")
for i = 1:length(params)
    vars(i) = var(endParams(:, i));
    means(i) = mean(endParams(:, i));
    fprintf(names(i));
    fprintf(" (expected=%.3f)=%.3f+-%.3f\n", params_ref(i), means(i), 3*sqrt(vars(i)))
end

function [ret] = normpdf_offset(x, mu, sigma, offset)
    ret = 1/(sigma*sqrt(2*pi))*exp(offset-0.5*((x-mu)/sigma).^2);
end

function [ret] = logp(x, mu, sigma)
    ret = -log(sigma)-0.5*(log(2*pi)+((x-mu)/sigma).^2);
end


function [ret] = f(x, a, t)
    ret = a*x;
end

function [ret] = df(x, a, t, isx0)
    ret = zeros(1, 4);
    %ret = (a+b*(1+x.^2).^-1).*x+c*cos(1.2*t);
    ret(1) = x;
end

function [ret] = ddf(x, a, t, isx0)
    ret = zeros(1, 1);
end

function [ret] = g(x, a, t)
    ret = x;
end

function [ret] = dg(x, a, t, isx0)
    ret = zeros(1, 4);
end

function [ret] = ddg(x, a, t, isx0)
    ret = zeros(4, 4);
end

function [QM] = Q(params, wit, wijt, swit, xTildes, y)
    [a] = toVals(params);
    q = 0.005; r = 0.01; x0 = 16;
    [M, N] = size(wit);
    %% Calculate I1, I2 and I3
    %I1 = sum(swit(:, 1).'.*log(normpdf(xTildes(:, :, 1)-f(x0, a, b, c, d, 0), 0, sqrt(q))));
    I1 = sum(swit(:, 1).'.*logp(xTildes(:, :, 1)-f(x0, a, 0), 0, sqrt(q)));
    I3 = 0;
    for t = 1:N
        %I3 = I3 + sum(swit(:, t).'.*log(normpdf(y(t)-g(xTildes(:, :, t), a, b, c, d, t), 0, sqrt(r))));
        I3 = I3 + sum(swit(:, t).'.*logp(y(t)-g(xTildes(:, :, t), a, t), 0, sqrt(r)));
    end
    I2 = 0;
    for t = 1:(N-1)
        for i = 1:M
            I2 = I2 + sum(wijt(i, :, t).*logp(xTildes(:, :, t+1)-f(xTildes(:, i, t), a, t), 0, sqrt(q)), 2);
        end
    end
    QM = I1 + I2 + I3;
end

function [wit, xTildes] = particleFilter(N, M, y, params)
    %% particle filter
    [a] = toVals(params);
    q = 0.005; r = 0.01; x0 = 16;
    xit = zeros(1, M, N);
    wit = zeros(M, N);
    %wit2 = zeros(M, N);
    % start
    xTilde = zeros(1, M);
    xTildes = zeros(1, M, N);
    for t=1:N
        % generate new particles
        if t~=1
            xTilde = f(xit(:, :, t-1), a, t-1)+sqrt(q)*randn(size(xTilde));
        else
            % Initialize particles x^i_0
            xi0 = x0+sqrt(q)*randn(size(xTilde));
            xTilde = f(xi0, a, 0)+sqrt(q)*randn(size(xTilde));
        end
        offset = min(((y(t)-g(xTilde, a, t)).^2))/(2*r);
        wit(:, t) = normpdf_offset(y(t)-g(xTilde, a, t), 0, sqrt(r), offset);
        soe = sum(wit(:, t));
        wit(:, t) = wit(:, t)/soe;
        xit(:, :, t) = randsample(xTilde, M, true, wit(:, t));
        xTildes(:, :, t) = xTilde;
    end
end

function [swit] = particleSmoother(N, M, wit, xTildes, params)
    %% particle smoother
    [a] = toVals(params);
    q = 0.005;
    swit = zeros(M, N);
    swit(:, N) = wit(:, N);
    for t=(N-1):-1:1
        indices = wit(:, t) > 1e-10;%wit(:, t) ~= 0;
        if sum(indices) == 0
            continue
        end
        ids = 1:M;
        kmask = ids(swit(:, t+1) > 1e-10);%swit(:, t+1) ~= 0);
        xmat = xTildes(:, kmask, t+1).'-f(xTildes(:, indices, t), a, t);
        offset = min(((xmat).^2), [], 2)/(2*q);
        vk = sum(repmat(wit(indices, t).', size(xmat, 1), 1).*normpdf_offset(xmat, 0, sqrt(q), offset), 2);
        for i = ids(indices)
            value = swit(kmask, t+1).'*(normpdf_offset((xTildes(:, kmask, t+1)-f(xTildes(:, i, t), a, t)).', 0, sqrt(q), offset)./vk);
            swit(i, t) = wit(i, t)*value;
        end
    end
end

function [a] = toVals(params)
    a = params(1);
end