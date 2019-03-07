function [EA,EC,EF,EN,EL] = calcLoveConstants(C) 

EA = C(1,1,1,1);
EC = C(3,3,3,3);
EF = C(1,1,3,3);
EN = C(1,2,1,2);
EL = C(1,3,1,3);