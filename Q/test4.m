clear all
close all
clc

load("thershold1001.mat")
plot(y4,'-o','MarkerIndices',1:15:length(y4),'MarkerSize',7);
hold on
plot(y3,'-o','MarkerIndices',1:15:length(y3),'MarkerSize',7);
plot(y2,'-o','MarkerIndices',1:15:length(y2),'MarkerSize',7);
plot(y1,'-o','MarkerIndices',1:15:length(y1),'MarkerSize',7);
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
legend('\epsilon= 1 bps','\epsilon= 100 bps','\epsilon= 200 bps','\epsilon= 300 bps','FontSize',15,'Location','NE');
%legend('\gamma= 0.4','\gamma= 0.5','\gamma= 0.6','\gamma= 0.7','FontSize',15,'Location','NE');
%legend('\alpha= 0.7','\alpha= 0.8','\alpha= 0.9','\alpha= 1.0','FontSize',15,'Location','SE');