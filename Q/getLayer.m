function Layer = getLayer(AdjMat, D, layerD)
N = size(AdjMat,1);
Layer = zeros(N,1);
for i = 1:N
    Layer(i) = floor(AdjMat(D,i)/layerD) + 1;
    if i == D
        Layer(i) = 0;
    end
end
end

