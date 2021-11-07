function [] = plotTFComparison(ref, A, B, C, D, mainTitle, ioMap)
    if ~exist('ioMap', 'var')
        ny = size(ref.C, 1);
        nu = size(ref.B, 2);
        ioMap = ones(nu, ny);
    end
    ioMap = logical(ioMap);
    us = sum(ioMap);
    yFlags = us~=0;
    for yId=1:length(yFlags)
        if ~yFlags(yId)
            continue
        end
        figure
        title([mainTitle ': Output ' num2str(yId)])
        col=1;
        for uId = 1:us(yId)
            if ~ioMap(uId, yId)
                continue
            end
            [magRef, phaseRef, wRef] = dbode(ref.A, ref.B, ref.C, ref.D, ref.Ts, uId);
            [mag, phase, w] = dbode(A, B, C, D, ref.Ts, uId);
            subplot(2, us(yId), col);
            plot(wRef, db(magRef))
            hold on
            plot(w, db(mag))
            xlabel('\omega (rad/s)')
            ylabel('Magnitude')
            
            subplot(2, us(yId), col+us(yId));
            plot(wRef, phaseRef)
            hold on
            plot(w, phase)
            xlabel('\omega (rad/s)')
            ylabel('Phase (degrees)')
            title(['Y' num2str(yId) '/U' num2str(uId)])
            col = col+1;
        end
    end
end