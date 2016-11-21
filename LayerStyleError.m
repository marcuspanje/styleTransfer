function err = LayerStyleError(G, A, nParams)
    diff = gather(G - A);

    [err, ~] = sumsqr(diff);

    err = err / (4 * nParams^2);
end
