clear all; clc;

node1=[4 5 6 7 8 9 10 11];
figure
%% time
colororder({'k','k'})%
time_FG = [2.99128736e-04,2.89197124e-04,3.82444135e-04,3.56986127e-04,3.01092737e-04,3.69781623e-04,-0.00126,0.002182974776];
time_FG1 = time_FG+0.00126;
time_NoFG = [0.000385649254,0.000689846732,0.005367674882,0.068712574859,1.453256663864,35.981267648127,0,100];
y = [time_FG1',time_FG',time_NoFG'];
b = bar(node1,y,1); hold on;
set(gca,'YScale','log');
b(1).FaceColor = 'k';
b(2).FaceColor = [0.75 0.75 0.75];
b(3).FaceColor = 'w';
xlabel('No. of nodes','FontSize',15);
ylabel('Computation time (s)','FontSize',15)
set(gca, 'xticklabels', {'4', '5', '6', '7', '8', '9',' ','50'}, 'Fontsize', 15);
axis([3.5 11.5 0.0001 100]);
text(9.8,200,'4.22\times10^7s','FontSize',15)
pbaspect([3 4 1])
uistack(b,'bottom');
grid on
legend('Original','FG without NLC','FG','FontSize',15,'Location','NW');
hold off