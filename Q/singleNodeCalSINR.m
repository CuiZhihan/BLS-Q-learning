function [MRe,C,C1,RouteAll,TimeSlot,Time,Q,SINR] = singleNodeCalSINR(resultMode,Loc)

global N SNRMat layerD PawgndB eta B L FixLayer

C = []; C1 = []; TimeSlot = []; Time = [];

%% calculate ALM SNR 计算邻接矩阵、ALM和SNR矩阵
AdjMat = zeros(N,N); ALMMat = zeros(N,N); SNRMat = zeros(N,N);PMat = zeros(N,N);
for i = 1:N
    for  j = i+1:N
        dist = sqrt(sum((Loc(i,:) -  Loc(j,:)).^2));
        AdjMat(i,j) = dist;
        [ALMMat(i,j), SNRMat(i,j), PMat(i,j)] = CalculateALM(dist);
    end
end
AdjMat = AdjMat + AdjMat';
ALMMat = ALMMat + ALMMat';
SNRMat = SNRMat + SNRMat';
PMat = PMat + PMat';


%% Get root(FG)
root=FGroot(ALMMat,N);
% disp(['root = ' num2str(root)])
% hold on
% plot(Loc(root,1),Loc(root,2),'o','MarkerSize',10,'MarkerFaceColor',[0,191,255]/255)

%% layering
Layer = getLayer(AdjMat, root, layerD);

%% sort nodes 根据到目的节点的距离进行排序（从大到小）
[~,Ind] = sort(AdjMat(:,root),'descend');

%% do DQN
SearchedNode = [];
RouteAll = {};

if  resultMode == 1
    NN = 1;
else
    NN = length(Ind) - 1;
end

for i = 1:NN
    if ~isempty(find(SearchedNode == Ind(i),1)) % searched
        continue;
    end
    
    if Layer(Ind(i)) == 1
        RouteAll = [RouteAll [Ind(i) root]];
        continue;
    end
    
       
    [MRe,Route,Q] = DQNFun(Ind(i),root,Layer);
    
    for j = 1:length(Route)
        if isempty(find(SearchedNode == Route(j)))
            SearchedNode = [SearchedNode Route(j)];
        end
    end
    
    RouteAll = [RouteAll Route];
end

% disp(MRe(end))
% 
% disp(Route)

if resultMode == 1
    return;
end

%train end here!


[TimeSlot,Time,SINR] = gertTimeSlotAndTimeSINR(RouteAll,PMat,2);
SINR = SINR;

%% capacity
C = (N-1)*L/sum(Time);

if resultMode == 3
    % no CoF 计算非CoF模式的容量
    [TimeSlot1,Time1] = gertTimeSlotAndTime(RouteAll,PMat,1);
    C1 = (N-1)*L/sum(Time1);
    return;    
end


%% graph
figure
ind = unique(Layer)';
colorData = rand(max(length(ind),length(RouteAll))+1,3);
for i = ind 
    idx = find(Layer == i);
    if i == 0
        plot(Loc(idx,1),Loc(idx,2),'p', 'MarkerSize',12, 'MarkerFaceColor','r','MarkerEdgeColor','r')
        hold on
    else
        plot(Loc(idx,1),Loc(idx,2),'o', 'MarkerSize',8, 'MarkerFaceColor',colorData(i,:),'MarkerEdgeColor',colorData(j,:))
    end
end

for i = 1:length(RouteAll)
    for j = 1:length(RouteAll{i}) - 1
        s = RouteAll{i}(j); d = RouteAll{i}(j+1);
        PlotLineArrow(gca, [Loc(s,1), Loc(d,1)], [Loc(s,2), Loc(d,2)], colorData(i,:));
    end
end
%saveas(gca,['Tree.fig'])
        
%% 分时隙绘图
% for i = 1:length(TimeSlot)
%     figure
%     for j = ind 
%         idx = find(Layer == j);
%         if j == 0
%             plot(Loc(idx,1),Loc(idx,2),'p', 'MarkerSize',12, 'MarkerFaceColor','r','MarkerEdgeColor','r')
%             hold on
%         else
%             plot(Loc(idx,1),Loc(idx,2),'o', 'MarkerSize',8, 'MarkerFaceColor',colorData(j,:),'MarkerEdgeColor',colorData(j,:))
%         end
%     end
%     
%     for j = 1:size(TimeSlot{i},1)
%         s = TimeSlot{i}(j,1); d = TimeSlot{i}(j,2);
%         PlotLineArrow(gca, [Loc(s,1), Loc(d,1)], [Loc(s,2), Loc(d,2)], 'r')
%     end
%     title(['TimeSlot=' num2str(i)])
%     
%     saveas(gca,['分时隙路径 TimeSlot=' num2str(i) '.fig'])
% end

end

