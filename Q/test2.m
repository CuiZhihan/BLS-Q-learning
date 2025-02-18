clear all
close all
clc

dist = 0:1:1000;
B = 20;   %MHz
PtxdB = 19;  % all dBm   802.11ax
sigma = 1;
PawgndB = -174;
z=raylrnd(sigma,1,1);
Ptx=10^((PtxdB-30)/10); %W
Pawgn=10^((PawgndB-30)/10); %W
ca = [];
for i = 1:1:length(dist);
G=pathloss_log3(dist(i),z,sigma);
S=Ptx*G;
SNR=S/(Pawgn*20000000);
c=B*(log2(1+SNR));
ca=[ca,c];
end
plot(dist,ca);
