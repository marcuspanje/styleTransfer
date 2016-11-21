function error = LayerStyleError(G, A, N, M)

    [error, ~] = sumsqr(G - A);

    error = error / (4 * N^2 * M^2);
end