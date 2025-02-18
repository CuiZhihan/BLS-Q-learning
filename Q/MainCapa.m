function [CResult]=MainCapa

%rand('state',3)

global alpha gamma epsion SNRMat N layerD PawgndB eta B L FixLayer

%% Paramenters
dataMode = 2; 
resultMode = 3; % 1-DQN 2-Tree、时隙图 3-Capacity
N = 20; % No. of node
maxNode = 100; % Max node（生成节点位置用）
B = 20000000; % bandwith mbps
L = 1000; % data packet
maxD = 600; % range
layerD = 110; % layer distance
alpha = 0.9; % learning rate
gamma = 0.5; % discount factor
epsion = 1; %convergence threshold bps
PawgndB = -174;
eta=10^((PawgndB-30)/10); %W

%% position 生成位置
if dataMode == 1
    load LocAll.mat
else
    LocAll = GenRandLoc(maxNode,maxD);
end


if resultMode == 1
    figure
    for N = [20 30 40]       
        Loc = LocAll(1:N,:);
        [MRe,~,~,~,~,~,Q] = singleNodeCal(resultMode,Loc);
        plot(MRe,'-o')
        hold on
    end
    hold off
    xlabel('number of itertatin')
    ylabel('Avg E2R Rate')
    legend('N=20','N=30','N=40')
    %saveas(gca,'compareN.fig')
elseif resultMode == 2
    N = 20;
    Loc = LocAll(1:N,:);
    [~,C,~,RouteAll,TimeSlot,Time] = singleNodeCal(resultMode,Loc);
elseif resultMode == 3
    NMat = [10:10:110];
    CResult = [];
    for N = NMat
        Loc = LocAll(1:N,:);
        temp = [];
%         for i = 1:2
        [~,C,C1,~,~,~] = singleNodeCal(resultMode,Loc);
%             temp = [temp; [C1 C]];
%         end
                
        CResult = [CResult [C1;C]];
    end
    
    figure
    plot(NMat,CResult(1,:),'-s')
    hold on
    plot(NMat,CResult(2,:),'-d')
    xlabel('Node Num')
    ylabel('Capacity')
    legend('DRL','DRL+CoF')
    
    %saveas(gca,'capacity.fig')
    
end
   