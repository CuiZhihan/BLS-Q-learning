clear all 
close all
clc

%rand('state',3)

global alpha gamma epsion SNRMat N layerD PawgndB eta B L FixLayer

%% Paramenters
dataMode = 2; 
resultMode = 3; % 1-DQN 2-Tree、时隙图 3-Capacity 4-MIPS
N = 20; % No. of node
maxNode = 130; % Max node（生成节点位置用）
B = 20000000; % bandwith mbps
L = 8000; % data packet
maxD = 500; % range
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
    for N = [50]       
        Loc = LocAll(1:N,:);
        [MRe,~,~,~,~,~,SINR] = singleNodeCalSINR(resultMode,Loc);
        plot(MRe,'-o')
        hold on
        [MRe1,~,~,~,~,~,~] = singleNodeCalMIPS(resultMode,Loc,SINR);
        plot(MRe1,'-o')
        hold on
    end
    hold off
    xlabel('Number of Iteration','FontSize',20)
    ylabel('Total Reward','FontSize',20)
    legend('DRL','MIPS')
%     xlim([0 320]);
%     ylim([3.2 4.2]);
    pbaspect([16 9 1])
    legend('DRL','MIPS','FontSize',20,'Location','SE');
    %saveas(gca,'compareN.fig')
elseif resultMode == 2
    N = 20;
    Loc = LocAll(1:N,:);
    [~,C,~,RouteAll,TimeSlot,Time] = singleNodeCal(resultMode,Loc);
elseif resultMode == 3
    NMat = [50:10:100];
    CResult = [];
    CResult2 = [];
    for N = NMat
        Loc = LocAll(1:N,:);
        temp = [];
        [~,C,C1,~,~,~,~,SINR] = singleNodeCalSINR(resultMode,Loc);
        CResult = [CResult [C1;C]];
        [~,C2,C3,~,~,~,~] = singleNodeCalMIPS(resultMode,Loc,SINR);
        CResult2 = [CResult2 [C3;C2]];

    end
    
    figure
    plot(NMat,CResult(1,:),'-s')
    hold on
    plot(NMat,CResult(2,:),'-d')
    plot(NMat,CResult2(1,:),'-o')
    plot(NMat,CResult2(2,:),'-x')
    xlabel('No. of Nodes')
    ylabel('Capacity')
    legend('DRL','DRL+CoF','MIPS','MIPS+CoF')
    
    %saveas(gca,'capacity.fig')
    
end
