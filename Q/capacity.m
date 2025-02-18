function [c]=capacity(B,SINR)
    c=B*(log2(1+SINR));
end