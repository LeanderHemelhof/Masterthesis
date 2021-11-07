function [x0, Lambda] = estimateInitialState(A, B, C, D, u, y, amt, x0Flag, Q, R)
    [n, nu] = size(B);
    ny = size(C, 1);
    weighted = exist('Q', 'var') ~= 0;
    nn = max(amt, n);
    Ys = y(:, 1:nn);
    Ys = Ys(:);
    Us = u(:, 1:nn);
    Us = Us(:);
    O = [];
    F = [];
    %Us = zeros(nu, 1);
    COV = zeros(nn*ny);
    for t=1:nn
        %Ys = [Ys;y(:, t)];
        %Us = [Us; u(:, t)];
        O = [O;C*A^t];
        tmp = [D zeros(ny, nu*(nn-t))];
        tmpCov = zeros(ny, ny);
        for i=t-2:-1:0
            CA = C*A^(t-i-2);
            tmp = [CA*B tmp];
            if weighted
                tmpCov = tmpCov + CA*Q*CA.';
            end
        end
        F = [F;tmp];
        if weighted
            COV(ny*(t-1)+(1:ny), ny*(t-1)+(1:ny)) = tmpCov+R;
        end
    end
    assert(rank(O) == n);
    % Ys=O*mu+F*Us => (Ys-F*Us)=O*mu
    %mu2 = O\(Ys-F*Us);
    if weighted
        W = inv(COV);
        x0 = (O.'*W*O)\O.'*W*(Ys-F*Us);
        Lambda = inv(O.'/COV*O);%pinv(O)*COV*pinv(O).';
    else
        x0 = O\(Ys-F*Us);
    end
    if ~x0Flag
        % x1 is wanted instead of x0
        % Assume no noise from x0 to x1
        x0 = A*x0;
    end
end