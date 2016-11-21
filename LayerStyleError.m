function error = LayerStyleError(G, A, N, M)
    diff = G - A;

    [error, ~] = sumsqr(diff);

    error = error / (4 * N^2 * M^2);
end