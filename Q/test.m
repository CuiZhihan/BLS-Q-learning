clear all
close all
% load("test3.mat");
timess = 1;
CResult = zeros(2,6);
CResult2 = zeros(2,6);
Tn = zeros(1,6);
Tm = zeros(1,6);
FGtime = zeros(1,6);
ss = 2;
for i = 1:1:ss
    [x,y,tn,tm,fgtime] = MainMIPS;
    CResult = CResult + x;
    CResult2 = CResult2 + y;
    Tn = Tn + tn;
    Tm = Tm + tm;
    FGtime = FGtime + fgtime;
    timess = timess+1
    %save("test3.mat","CResult","CResult2","Tn","Tm","FGtime","timess");
end


