from createCubicElasticityMatrix import *
from array_utils import print_cs
import numpy as np
import matplotlib.pyplot as plt

c11 = 1405.9  * 1.e9 # [Pa]
c12 = 1364.8  * 1.e9 # [Pa]
# NOTE: Taku's Values are given in Voight Notation, so there is an extra factor
#       of 2 in definition of c44: \hat{c44} = 2*c44 --> c44 = \hat{c44}/2
c44_hat =  397.9  * 1.e9 # [Pa]
c44 = c44_hat/2
rho =   12.98 * 1.e3 # [kg/m^3]

Pressure =  357.5 * 1.e9 # [Pa]
Temperature = 6000 # [K]

# Create Elasticity 4-Tensor
C = create_cubic_elasticity_matrix(c11, c12, c44)
fig = plot_cubic_wavespeeds(c11, c12, c44, rho)
plt.savefig('../../analaytic_cubic_wavespeeds.pdf')
plt.close()

fig = plot_wavespeeds(*get_acoustic_Pwavespeeds(C, rho))
plt.savefig('../../numerical_cubic_wavespeeds.pdf')
plt.close()

# Display the non-zero components
print("Original Tensor in 6-space Tensor Notations:")
M = getHookeLawMatrix(C)

# Create rotation matrix
R = rotation_matrix(np.pi/4, -55.*np.pi/180)

#TODO: Add test to suite!
# Rotate Elasticity Tensor
Cprime = transform_tensor(C, R)
fig = plot_wavespeeds(*get_acoustic_Pwavespeeds(Cprime, rho))
plt.savefig('../../rotated_cubic_wavespeeds.pdf')
plt.close()





# v = np.array([1, 1, 1])
# s =1./np.sqrt(2)
# a_ij = np.array([[s, 0, -s],[0, 1, 0],[s, 0, s]])
# R = rotation_matrix(0, np.pi/4)
# print_cs(R.T)
# Cprime = transform_tensor(v, R.T)
# print_cs(Cprime)
#
# Cprime2 = transform_tensor(v, a_ij)
# print_cs(Cprime2)
# t_ij = np.array([[2,6,4], [0, 8, 0], [4, 2, 0]])
# Cprime3 = transform_tensor(t_ij, a_ij)
# print_cs(Cprime3)
# Cprime2 = transform_tensor(t_ij, R.T)
# print_cs(Cprime2)
