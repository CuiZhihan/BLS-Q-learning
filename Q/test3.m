clear all
close all
clc

x=[0.1 0.9 2.1 3.2 4.0 4.8 5.7 7.0 8.2 9.1 10.0];
y=[0.0105 0.8505 4.6305 10.752 16.800 24.192 29.640 ...
    36.400 42.640 47.320 52.000];
p=polyfit(x,y,1);
f=polyval(p,x);
plot(x,y,'o','MarkerSize',10);
hold on
plot(x,f,'-','linewidth',2);
set(gca,'FontSize',15)
legend('Raw data','Line','FontSize',20,'Location','SE');
xlabel('X','FontSize',20);
ylabel('Y','FontSize',20);
