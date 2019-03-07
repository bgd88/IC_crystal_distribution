% Inner Core values from Taku
c11 = 1405.9  * 1.e9; % [Pa]
c12 = 1364.8  * 1.e9; % [Pa]
c44 =  397.9  * 1.e9; % [Pa]
rho =   12.98 * 1.e3; % [kg/m^3]

Pressure =  357.5 * 1.e9; % [Pa]
Temperature = 6000; % [K]

% Create Elasticity 4-Tensor
C = createCubicElasticityMatrix(c11, c12, c44);

% Display the non-zero components 
M = displayHookeLawMatrix(C)

% Calculate Isotropic Body-Wave Speeds
[vP, vS] = calcBodywaveSpeeds(C, rho)

% Plot P-wavespeed as function of Prop. 
[phi, theta, v] = longitudinalCubicWavespeeds(c11, c12, c44, rho);

% Create Rotation Matrix 
% 45 degrees anit-clockwise about Z-axis
Rz = makeEulerRotation(pi/2, 0 , 0);
% 45 degrees anit-clockwise about Y-axis
Ry = makeEulerRotation(0, pi/4 , 0);
Rtest = makeEulerRotation(pi/4, pi/4 , 0);

Rz2 = makeAngleAxisRotation(-45, [0,0,1]);
Ry2 = makeAngleAxisRotation(-45, [0,1,0]);
Rtest2 = Ry2*Rz2;

% Rotate Tensor
Cnew = transform_tensor(C, Rz);