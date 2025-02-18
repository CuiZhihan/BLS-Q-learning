function G = pathloss_log3(dist,z,sigma)
    % Define and initialize the parameters that used in this function
    d0 = 10.0; % meter
    alpha = 3.5;
    wall = 0.0;
    X = 4.0; % dB
    PL0 = 20.0*log10(d0);
    PL1 = PL0 + (10.0*alpha*log10(dist/d0)) - wall + X;
    G = 1.0/(10.0)^(PL1/10.0);
end