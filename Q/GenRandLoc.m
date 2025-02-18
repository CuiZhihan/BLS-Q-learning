function [LocAll,LayerAll] = GenRandLoc(maxNum,maxD)
global layerD

LocAll = [];
for i = 1:floor(maxNum/10)+1
    while true
        Loc = [];
        for j = 1:10
            while true
                x = randperm(maxD-2,1);
                y = randperm(maxD-2,1);
                flag = 0;
                for j = 1:size(Loc,1)
                    % no same position 保证节点不相等
                    if x == Loc(j,1) && y == Loc(j,2)
                        flag = 1;
                    end
                end
                if flag
                    continue;
                end
                Loc = [Loc; [x y]];
                break;
            end
        end
        
        %% no cross layer 保证不跨层        
        N = size(Loc,1);
        
        %% calculate ALM SNR
        AdjMat = zeros(N,N); %distance
        ALMMat = zeros(N,N); 
        for i = 1:N
            for  j = i+1:N
                dist = sqrt(sum((Loc(i,:) -  Loc(j,:)).^2));
                AdjMat(i,j) = dist;
                ALMMat(i,j) = CalculateALM(dist);
            end
        end
        AdjMat = AdjMat + AdjMat';
        ALMMat = ALMMat + ALMMat';

        %% Get root
        root=FGroot(ALMMat,N);

        %% layering
        Layer = getLayer(AdjMat, root, layerD);
        
        if sum(unique(Layer)) ==sum(0:max(Layer))
            LocAll = [LocAll; Loc];
            break;
        end
    end
end

N = size(LocAll,1);

%% 计算邻接矩阵、ALM和SNR矩阵
AdjMat = zeros(N,N); ALMMat = zeros(N,N); 
for i = 1:N
    for  j = i+1:N
        dist = sqrt(sum((LocAll(i,:) -  LocAll(j,:)).^2));
        AdjMat(i,j) = dist;
        ALMMat(i,j) = CalculateALM(dist);
    end
end
AdjMat = AdjMat + AdjMat';
ALMMat = ALMMat + ALMMat';

%% 确定目的节点(FG)
root=FGroot(ALMMat,N);

%% 分层
LayerAll = getLayer(AdjMat, root, layerD);


%save LocAll.mat LocAll LayerAll
end



