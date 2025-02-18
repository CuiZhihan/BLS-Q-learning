clear all
close all
clc

NMat = 50:10:100;
load("test6.mat")
% load("MILPSCa.mat")
CResult = CResult/(timess*1000000);
CResult2 = CResult2/(timess*1000000);
FGResult = FGResult/(timess*1000000);
figure
plot(NMat,CResult2(2,:),'r-o','linewidth',2.5,'MarkerSize',9)

hold on
plot(NMat,CResult2(1,:),'r--x','linewidth',2.5,'MarkerSize',9)
plot(NMat,CResult(2,:),'b-o','linewidth',2.5,'MarkerSize',9)
plot(NMat,CResult(1,:),'b--x','linewidth',2.5,'MarkerSize',9)
plot(NMat,FGResult,'k-o','linewidth',2.5,'MarkerSize',9)
xlabel('No. of nodes','FontSize',20)
ylabel('Average network capacity (Mbps)','FontSize',20)
ylim([0 7]);
xlim([50 100]);
xticks([50 60 70 80 90 100]);
% yticks([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8]);
set(gca,'FontSize',15)
set(gca,'LooseInset',get(gca,'TightInset'))
pbaspect([3 4 1])
legend('INLPS','INLPS without NLC','NLPS','NLPS without NLC','FG','FontSize',15,'Location','NW','EdgeColor',[1 1 1])
dim = [.3 .61 .3 .3];
str = {'\alpha = 0.9','\gamma = 0.5','\epsilon = 1 bps'};
annotation('textbox',dim,'String',str,'FitBoxToText','on',...
    'FontSize',15,'BackgroundColor',[1 1 1],'EdgeColor',[1 1 1]);

grid on;