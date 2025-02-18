clear all
close all
clc
% load("MIPStraingraph.mat")
load("DQN2.mat")
x1 = [101 101];
y1 = [0 3.3548];
x2 = [3.3548 3.3548];
y2 = [0 101];
x11 = 101;
y11 = 3.3548;

x3 = [185 185];
y3 = [0 3.5935];
x4 = [3.5935 3.5935];
y4 = [0 185];
x33 = 185;
y33 = 3.5935;

figure
plot(MRe1,'r-o','MarkerIndices',1:10:length(MRe),'MarkerSize',7)
hold on
plot(MRe,'b-o','linewidth',1,'MarkerIndices',1:10:length(MRe),'MarkerSize',7)
plot(x1,y1,'k--','linewidth',1);
plot(y2,x2,'k--','linewidth',1);
plot(x3,y3,'k--','linewidth',1);
plot(y4,x4,'k--','linewidth',1);
scatter(x11,y11,50,'k');
scatter(x33,y33,50,'k');
hold off
xlabel('No. of iterations','FontSize',20)
ylabel('Total reward','FontSize',20)
xlim([0 310]);
ylim([2.6 3.7]);
xticks([0 50 101 150 185 250 300]);
% xtickangle(45);
yticks([2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.354 3.4 3.5 3.593 3.7]);
set(gca,'FontSize',15)
pbaspect([3 4 1])
legend('MILPS','NILPS','FontSize',15,'Location','SE');
% grid on