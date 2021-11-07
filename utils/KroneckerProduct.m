function [mat] = KroneckerProduct(A, B)
    [Ah, Aw] = size(A);
    [Bh, Bw] = size(B);
    mat = zeros(size(B).*size(A));
    rowOff = 0;
    for y=1:Ah
        colOff = 0;
        for x=1:Aw
            mat(rowOff+(1:Bh), colOff+(1:Bw)) = A(y, x)*B;
            colOff = colOff+Bw;
        end
        rowOff = rowOff+Bh;
    end
end