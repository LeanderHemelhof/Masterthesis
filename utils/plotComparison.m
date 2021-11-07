function [] = plotComparison(yref, uref, A, B, C, D, K, mainTitle, range, outputs)
    x1 = estimateInitialState(A, B, C, D, uref, yref, 20, false);
    ysim = dlsim(A-K*C, [B-K*D K], C, [D zeros(size(D, 1))], [uref.' yref.'], x1).';
    if ~exist('range', 'var')
        range = 1:size(yref, 2);
    end
    if ~exist('outputs', 'var')
        outputs = 1:size(yref, 1);
    end
    figure
    title(mainTitle)
    if length(outputs) > 1
        rows = 2;
    else
        rows = 1;
    end
    cols = ceil(size(outputs, 2)/rows);
    for i=1:size(outputs, 2)
        subplot(rows, cols, i)
        hold on
        plot(range, yref(outputs(i), range), 'k:')
        plot(range, ysim(outputs(i), range), 'r--')
        title(['Output ' num2str(outputs(i))])
        legend('Experimental measurements', 'Simulated outputs using same inputs')
    end
end