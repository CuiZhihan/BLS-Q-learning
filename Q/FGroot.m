function root=FGroot(ALM,nodeNums)
min_g = inf;
    min_FG = [];
    root = 0;
    for n = 1:nodeNums
        sp = n;                     % root
        [FG] = Dijkstra(ALM,sp);    
        [g] = CalculateG(FG,ALM);   
        if g < min_g
            min_g = g;
            min_FG = FG;
            root = sp;
        end
    end

end
