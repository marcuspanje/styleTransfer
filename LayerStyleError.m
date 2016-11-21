function error = LayerStyleError(G, A, N, M)

    error = 0;
    
    for r = 1 : size(G,1)
        for c = 1 : size(G,2)
            error = error + (G(r,c) - A(r,c))^2;
        end
    end

    error = error / (4 * N^2 * M^2);
end