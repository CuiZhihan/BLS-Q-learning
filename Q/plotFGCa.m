clear all; 
close all
clc;
load("FGCa.mat");
figure
colororder({'k','k'})
x = [1 2 3];
y = [noFG',FG',FGNLC'];
y = y/(timess*1000000);
b = bar(x,y,0.7); hold on;
b.FaceColor = 'flat';
b.CData(1,:) = [1 1 1];
b.CData(2,:) = [0.75 0.75 0.75];
b.CData(3,:) = [0 0 0];
% set(gca,'YScale','log');
% b(1).FaceColor = 'k';
% b(2).FaceColor = [0.75 0.75 0.75];
% b(3).FaceColor = 'w';
set(gca,'FontSize',15);
set(gca,'XTickLabel',{'TBR','FG without NLC','FG'},'FontSize',10.5);
%xlabel('No. of nodes','FontSize',15);
ylabel('Network capacity (Mbps)','FontSize',15);
dim = [.32 .56 .3 .3];
%str2 = {'Network size is 500m\times500m'}
str = {'Network size is 500m\times500m','No. of simulations is 10000','Average no. of hops is 2.3','Transmit power is 19 dBm','No. of nodes is 50'};
%annotation('textbox',dim,'String',str2,'FitBoxToText','on','FontSize',15,'BackgroundColor',[1 1 1],'EdgeColor',[1 1 1]);
annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',13,'BackgroundColor',[1 1 1],'EdgeColor',[1 1 1]);
%text(0.8,0.045,'100%','FontSize',15);
text(1.65,0.285,'9.0 times','FontSize',15);
text(2.6,0.41,'13.2 times','FontSize',15)
%uistack(b,'bottom');
%set(gca,'FontSize',15);
ylim([0 0.45]);
pbaspect([3 4 1]);
set(gca, 'YGrid', 'on', 'XGrid', 'off')
% grid on
% legend('Original','FG only','FG with NLC','FontSize',15,'Location','NW');
hold off



