import pandas as pd

# model3-50	      10	 21.25 	 363.8524	5000  12.457691
A0 = 1815.9e9
C0 = 1963.8e9
F0 = 1223.8e9
L0 =   79.8e9
N0 =   49.0e9
comp0 = [A0, C0, F0, L0, N0]

# Vocaldo 2009
#                 P(GPa)	T (K)  Density
# A1 Hcp	       293     	3000  13.32
A1 = 1915e9
C1 = 2109e9
L1 =   313e9
N1 =   396e9
F1 = 986e9
comp1 = [A1, C1, F1, L1, N1]

# Vocaldo 2009
#                 P(GPa)	T (K)  Density
# B Hcp	       308     	5000  13.15
A2 = 1689e9
C2 = 1725e9
L2 =   216e9
N2 =   252e9
F2 = 990e9
comp2 = [A2, C2, F2, L2, N2]

# Vocaldo 2018 - Full Set of triclinic Elastic Constants Projected onto Hexagonal
#                       P(GPa)	   T (K)  Density
# B Hcp-Fe60Si2C2 SQS	  360       6500  13.10
A3 = 1578e9
C3 = 1811e9
F3 = 1067e9
L3 =  146e9
N3 =  182e9
comp3 = [A3, C3, F3, L3, N3]

# Vocaldo 2018 - Hexagonal Symmetry
#                       P(GPa)	   T (K)  Density
# B Hcp-Fe60Si2C2 SQS	  360       6500  13.10
A = 1547e9
C = 1811e9
F = 1117e9
L =  179e9
N =  109e9
comp4 = [A, C, F, L, N]

# Estimated Temp. Corrections from Steinle-Neumann
# DO NOT USE IN ANY PUBLICATION!!!!!!
A = 1547e9 + 20e9
C = 1811e9 + 215e9
F = 1117e9 + 65e9
L =  179e9
N =  109e9
comp4b = [A, C, F, L, N]



# Li 2018 - Hexagonal Symmetry
#                       P(GPa)	   T (K)  Density
A = 1780e9
C = 1956e9
F = 1059e9
L = 237e9
N = 266.5e9
comp5 = [A, C, F, L, N]

# Steinle-Neumann 2000 - Hexagonal Symmetry
#                       P(GPa)	   T (K)  Density
# B Hcp-Fe  360        6000  13.49
A = 2150e9
C = 1685e9
F = 990e9
L = 140e9
N = 60e9
rho=13.04e3
comp6 = [A, C, F, L, N]

pure = pd.Series(comp6, index=['A','C','F','L','N'])
rho_pure = 13.04e3

FeSiNi  = pd.Series(comp0, index=['A','C','F','L','N'])
rho_FeSiNi = 12.457691e3

FeSiC  = pd.Series(comp4, index=['A','C','F','L','N'])
rho_FeSiC = 13.10e3

IMIC_prime = pd.Series(np.array([1676, 1840, 1196, 148, 126])*1.e9, index=['A','C','F','L','N'])
rho_IMIC = 13.1e3
