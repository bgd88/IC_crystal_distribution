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
M = displayHookeLawMatrix(C);

% Calculate Isotropic Body-Wave Speeds
[vP, vS] = calcBodywaveSpeeds(C, rho);