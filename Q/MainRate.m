clear all
clc

% rand('state',3)

global alpha gamma epsion SNRMat N layerD PawgndB eta B L FixLayer

%% Paramenters
dataMode = 2; 
resultMode = 1; % 1-DQN 2-Tree、时隙图 3-Capacity 4-MIPS
N = 50; % No. of node
maxNode = 100; % Max node（生成节点位置用）
B = 20000000; % bandwith bps
L = 8000; % data packet  in bits
maxD = 1000; % range
layerD = 170; % layer distance
alpha = 1; % learning rate
gamma = 0.5; % discount factor
epsion = 1; %convergence threshold bps
PawgndB = -174;
eta=10^((PawgndB-30)/10); %W

%% position 生成位置
if dataMode == 1
    load Location.mat
else
    LocAll = GenRandLoc(maxNode,maxD);
end


if resultMode == 1
    Alpha = [0.7 0.8 0.9 1];
    Gamma = [0.4 0.5 0.6 0.7];
    Epsion = [1 100 200 300];
    figure
%     for alpha = Alpha
    %for gamma = Gamma
    for epsion = Epsion
    for N = 50       
        Loc = LocAll(1:N,:);
        [MRe,~,~,~,~,~,SINR] = singleNodeCalSINR(resultMode,Loc);
        plot(MRe,'-o','MarkerIndices',1:15:length(MRe),'MarkerSize',7)
        hold on
%         [~,MRe1,~,~,~,~,~,~] = singleNodeCalMIPS(resultMode,Loc,SINR);
%         plot(MRe1,'-o')
%         hold on
    end
    end
%     hold off
% plot(MRe1,'r-o','MarkerIndices',1:10:length(MRe),'MarkerSize',7)
% hold on
% plot(MRe,'b-o','linewidth',1,'MarkerIndices',1:10:length(MRe),'MarkerSize',7)
% plot(x1,y1,'k--','linewidth',1);
% plot(y2,x2,'k--','linewidth',1);
% plot(x3,y3,'k--','linewidth',1);
% plot(y4,x4,'k--','linewidth',1);
% scatter(x11,y11,50,'k');
% scatter(x33,y33,50,'k');
hold off
xlabel('No. of iterations','FontSize',20)
ylabel('Average E2E rate (bps)','FontSize',20)
xlim([0 310]);
ylim([2.2 3.7]);
% xticks([0 50 101 150 185 250 300]);
% xtickangle(45);
% yticks([2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.354 3.4 3.5 3.593 3.7]);
set(gca,'FontSize',15)
pbaspect([3 4 1])
grid on
%legend('\alpha= 0.7','\alpha= 0.8','\alpha= 0.9','\alpha= 1.0','FontSize',15,'Location','SE');
%legend('\gamma= 0.4','\gamma= 0.5','\gamma= 0.6','\gamma= 0.7','FontSize',15,'Location','SE');
legend('\epsilon= 1','\epsilon= 100','\epsilon= 200','\epsilon= 300','FontSize',15,'Location','SE');





%     xlabel('Number of Iteration','FontSize',20)
%     ylabel('Total Reward','FontSize',20)
%     legend('DRL')
%     xlim([0 320]);
%     ylim([3.2 4.2]);
%     pbaspect([16 9 1])
%     legend('DRL','FontSize',20,'Location','SE');
%     legend('DRL','FontSize',20,'Location','SE');
%     saveas(gca,'compareN.fig')
elseif resultMode == 2
    N = 50;
    Loc = LocAll(1:N,:);
    [~,C,~,RouteAll,TimeSlot,Time] = singleNodeCal(resultMode,Loc);
elseif resultMode == 3
   
    NMat = [50:10:100];
    CResult = [];
    CResult2 = [];
    Tm = [];
    Tn = [];
    FGtime =[];
    for N = NMat
        Loc = LocAll(1:N,:);
        temp = [];
        tic
        [~,C,C1,~,~,~,~,SINR] = singleNodeCalSINR(resultMode,Loc);
        CResult = [CResult [C1;C]];
        tn = toc;
        Tn = [Tn tn];
        tic
        [fgtime,~,C2,C3,~,~,~,~] = singleNodeCalMIPS(resultMode,Loc,SINR);
        CResult2 = [CResult2 [C3;C2]];
        tm = toc;
        Tm = [Tm tm];
        FGtime = [FGtime fgtime];

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