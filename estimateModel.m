function [A, B, C, D] = estimateModel(Gk, n, svThresh, qandr)
%estimateModel Estimates model based on DFT data
%   Extra parameters are largest allowable ratio between the largest and
%   smallest valid singular value. The second parameter is the values given
%   to q and r in the algorithm. Bigger values give better approximations,
%   but is bounded by q+r = 2*qandr <= 2*M = length(Gk)
    [p, m, M2] = size(Gk);
    M = M2/2;
    wks = (0:M)*pi/M;
    wksx = 0:(pi/M):2*pi;
    wksx(end) = [];
    q = qandr;
    r = qandr;
    his = zeros(p, m, q+r-1);
    is = 1:q+r-1;
    for k=0:(2*M-1)
        e = reshape(exp(1j*2*pi*is*k/(2*M)), 1, 1, []);
        his = his+repmat(Gk(:, :, k+1), 1, 1, q+r-1).*e;
%         for i=is
%             his(:, :, i) = his(:, :, i) + Gk(:, :, k+1)*exp(1j*2*pi*i*k/(2*M));
%         end
    end
    his = his/(2*M);
    H = zeros(q*p, r*m);
    rid = 1;
    for row = 1:q
        H(rid+(0:p-1), :) = reshape(real(his(:, :, row+(0:r-1))), p, []);
        rid = rid+p;
    end
    [U, S, ~] = svd(H);
    svs = diag(S);
%     n = 1;
%     rel = svs(n+1)/svs(1);
%     while rel > svThresh
%         n = n+1;
%         rel = svs(n+1)/svs(1);
%     end
%     assert(n == 3)
    Us = U(:, 1:n);
    J1 = [eye((q-1)*p) zeros((q-1)*p, p)];
    J2 = [zeros((q-1)*p, p) eye((q-1)*p)];
    J3 = [eye(p) zeros(p, (q-1)*p)];
    A  = real(pinv(J1*Us)*J2*Us);
    C  = J3*Us;
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
    D = BD(n+1:end, :);
end