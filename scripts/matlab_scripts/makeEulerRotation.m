% function R = makeEulerRotation(alpha, beta, gamma)
%
% Rotation Matrix Consistent with Dahlen and Tromp [1998]
%
% Make the 3x3 rotation matrix for the rotation with Euler angles: 
% alpha & gamma: [0..2*pi]
% beta: [0..pi]
% Edited: BD 2/11/19
function R = makeEulerRotation(alpha, beta, gamma)


cA = cos(alpha);
sA = sin(alpha);

cB = cos(beta);
sB = sin(beta);

cG = cos(gamma);
sG = sin(gamma);

R = [ cA*cB*cG-sA*sG,  sA*cB*cG+cA*sG, -sB*cG;
     -cA*cB*sG-sA*cG, -sA*cB*sG+cA*cG,  sB*sG;
               cA*sB,           sA*sB,    cB ];
   
end