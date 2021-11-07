function [Sxx, Sxb, Sbb, Syy, Syx, Syu, Sxu1, Sbu, Suu1, Sxu2, Suu2] = ExcitationStep(PNks, xNks, PNkkm1s, u, y)
    N = length(y)-1;
    %% Expectation step
    Sxx = zeros(size(PNks{1}));
    for k = 1:N
        Sxx = Sxx + PNks{k+1} + xNks(:, k+1)*xNks(:, k+1).';
    end
    Sxb = zeros(size(PNks{1}));
    for k = 1:N
        Sxb = Sxb + PNkkm1s{k} + xNks(:, k+1)*xNks(:, k).';
    end
    Sbb = zeros(size(PNks{1}));
    for k = 1:N
        Sbb = Sbb + PNks{k} + xNks(:, k)*xNks(:, k).';
    end
    Syy = zeros(size(y, 1));
    for k = 1:N
        Syy = Syy + y(k+1)*y(k+1).';
    end
    Syx = zeros(size(y, 1), size(xNks(:, 1), 1));
    for k = 1:N
        Syx = Syx + y(k+1)*xNks(:, k+1).';
    end
    Syu = zeros(size(y, 1), size(u, 1));
    for k = 1:N
        Syu = Syu + y(k+1)*u(k+1).';
    end
    Sxu1 = zeros(size(xNks(:, 1), 1), size(u, 1));
    for k = 1:N
        Sxu1 = Sxu1 + xNks(:, k+1)*u(k).';
    end
    Sbu = zeros(size(xNks(:, 1), 1), size(u, 1));
    for k = 1:N
        Sbu = Sbu + xNks(:, k+1)*u(k).';
    end
    Suu1 = zeros(size(u, 1));
    for k = 1:N
        Suu1 = Suu1 + u(k)*u(k).';
    end
    Sxu2 = zeros(size(xNks(:, 1), 1), size(u, 1));
    for k = 1:N
        Sxu2 = Sxu2 + xNks(:, k+1)*u(k+1).';
    end
    Suu2 = zeros(size(u, 1));
    for k = 1:N
        Suu2 = Suu2 + u(k+1)*u(k+1).';
    end
end

