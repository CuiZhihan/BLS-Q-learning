clear all; 
clear all;
close all;
clc;
figure
node1 = [50 60 70 80 90 100];
colororder({'k','k'})
% FG = [0.009168 0.010321 0.013572 0.017331 0.022997 0.029080];
% NILPS = [0.272300 0.360964 0.498692 0.650590 0.802378 1.025334];
% MILPS = [0.448388 0.662292 0.950735 1.258148 1.759559 2.118289];
load("test5.mat")
Tn = Tn/(timess);
Tm = Tm/(timess);
Tm = Tn + Tm;
FGtime = FGtime/(timess);
y = [FGtime',Tn',Tm'];
b = bar(node1,y,1); hold on;
b(1).FaceColor = 'k';
b(2).FaceColor = 'b';
b(3).FaceColor = 'r';
xlabel('No. of nodes','FontSize',20);
ylabel('Computation time (s)','FontSize',20);
% uistack(b,'bottom');
set(gca,'FontSize',15);
set(gca,'YScale','log');
ylim([0.001 100]);
% ylim([0 2.2]);
% yticks([0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2 2.2]);
xlim([45 105]);
pbaspect([3 4 1]);
grid on
legend('No DRL','NILPS','MILPS','FontSize',15,'Location','NW','EdgeColor',[1 1 1]);
dim = [.57 .61 .3 .3];
str = {'\alpha = 0.9','\gamma = 0.5','\epsilon = 1 bps'};
annotation('textbox',dim,'String',str,'FitBoxToText','on',...
    'FontSize',15,'BackgroundColor',[1 1 1],'EdgeColor',[1 1 1]);
% set(gca, 'YGrid', 'on', 'XGrid', 'off');
hold off
