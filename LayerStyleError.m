function err = LayerStyleError(G, A, nParams)
    diff = (G - A);

    [err, ~] = gather(sumsqr(diff));

    %err = err / (4 * nParams^2);
    err = err / nParams;
end
