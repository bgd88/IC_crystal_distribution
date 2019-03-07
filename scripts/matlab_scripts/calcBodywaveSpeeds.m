function [vP, vS] = calcBodywaveSpeeds(C, rho)

[EA,EC,EF,EN,EL] = calcLoveConstants(C); 

% Calculate Isotropic Body-Wave speeds for transversely Isotropic material.
% 
vP = sqrt((8*EA+3*EC+4*EF+8*EL)/(15*rho));
vS = sqrt((EA + EC - 2*EF + 6*EL + 5*EN)/(15*rho));