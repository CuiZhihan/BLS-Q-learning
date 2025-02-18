clc
clear
close all

load MRe_20 
load MRe_50 
load MRe_100

figure
plot(MRe_20,'-o')
hold on
plot(MRe_50,'-o')
plot(MRe_100,'-o')
xlabel('µü´ú´ÎÊý')
ylabel('Avg E2R Rate')
legend('N=20','N=50','N=100')