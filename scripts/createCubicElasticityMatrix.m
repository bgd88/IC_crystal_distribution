function C = createCubicElasticityMatrix(c11, c12, c44)

% Using Conventions detailed in "Eigentensors of linear Anisotropic
% Materials" by Mehrabadi & Cowin (1989). 

C = zeros(3,3,3,3);

C(1,1,1,1) = c11;
C(2,2,1,1) = c12;
C(3,3,1,1) = c12;

C(1,1,2,2) = c12;
C(2,2,2,2) = c11;
C(3,3,2,2) = c12;

C(1,1,3,3) = c12;
C(2,2,3,3) = c12;
C(3,3,3,3) = c11;
% 
% C(2,3,2,3) = 2*c44;
% C(1,3,1,3) = 2*c44;
% C(1,2,1,2) = 2*c44;
C(2,3,2,3) = c44;
C(1,3,1,3) = c44;
C(1,2,1,2) = c44;
