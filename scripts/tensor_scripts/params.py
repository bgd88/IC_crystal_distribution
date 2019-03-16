figDir = "../../figures/"
wDir = "../../IC_compositions/"

# Values from Taku
c11_hat = 1405.9  * 1.e9 # [Pa]
c12_hat = 1364.8  * 1.e9 # [Pa]
c44_hat =  397.9  * 1.e9# [Pa]
rho =   12.98 * 1.e3 # [kg/m^3]
Pressure =  357.5 * 1.e9 # [Pa]
Temperature = 6000 # [K]
# NOTE: Taku's Values are given in Voight Notation, so there is an extra factor
#       of 2 in definition of c44: \hat{c44} = 2*c44 --> c44 = \hat{c44}/2
FeSi_elastic_params = {'c11' : c11_hat,
                       'c12' : c12_hat,
                       'c44' : c44_hat/2}

# Inversion of A, C, F, L, N from Ishii 2002 & PREM constrains
prem_elastic_parms = {'A' : 1558.7*1.e9,
                      'C' : 1504.5*1.e9,
                      'F' : 1239.0*1.e9,
                      'L' :  168.3*1.e9,
                      'N' :  153.0*1.e9 }
