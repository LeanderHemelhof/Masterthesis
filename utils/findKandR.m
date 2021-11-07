function [K, Rr] = findKandR(A, C, Q, R, S)
    n = size(A, 1);
    tmp = 1.0001*eye(n*n)-KroneckerProduct(A, A);% Fix for pole at 1
    Sigmak = tmp\reshape(Q, [], 1);
    Sigmak = reshape(Sigmak, n, n);
    
    G = A*Sigmak*C.'+S;
    Lambda0 = C*Sigmak*C.'+R;
    
    Left = [A.'-C.'/Lambda0*G.' zeros(n);-G/Lambda0*G.' eye(n)];
    Right = [eye(n) -C.'/Lambda0*C;zeros(n) A-G/Lambda0*C];
    [V, EVs] = eig(Left, Right);
    [~, ids] = sort(abs(diag(EVs)));
    P = V(n+1:end, ids(1:n))/V(1:n, ids(1:n));
    assert(max(abs(imag(P)), [], 'all') < 1e-5)
    P = real(P);
    
    Rr = Lambda0 - C*P*C.';
    K = (G-A*P*C.')/Rr;
    %test = A*P*A.'+K*(G-A*P*C.').';
end