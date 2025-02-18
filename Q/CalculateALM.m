function [ALM,SNR,S] = CalculateALM(dist)
    DIFS = 34;
    SIFS = 16;
    PLCP = 39.2;
    MACheader = 53.3;
    CWmin = 15;
    ACK = 18.7;
    Timeslot = 9;   %all in us
    PawgndB = -174;
    PtxdB = 19;  % all dBm   802.11ax
    Bt = 8192; %bit
    TxR = 6; %Mbps
    sigma = 1;
    O = PLCP+DIFS+SIFS+MACheader+CWmin+ACK+Timeslot;
    z=raylrnd(sigma,1,1); 
    Pawgn=10^((PawgndB-30)/10); %W
    Ptx=10^((PtxdB-30)/10); %W

    G=pathloss_log3(dist,z,sigma);
    S=Ptx*G;
    BER=Pawgn/S;
    SNR=S/(Pawgn*20000000);
    FER=BER*Bt;
    ALM=(O+(Bt/TxR))/(1-FER);
end

