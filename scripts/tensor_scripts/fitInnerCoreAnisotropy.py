from createCubicElasticityMatrix import *
from array_utils import print_cs
import numpy as np

c11 = 1405.9  * 1.e9 # [Pa]
c12 = 1364.8  * 1.e9 # [Pa]
c44 =  397.9  * 1.e9 # [Pa]
rho =   12.98 * 1.e3 # [kg/m^3]

Pressure =  357.5 * 1.e9 # [Pa]
Temperature = 6000 # [K]

# Use Thresholds
rotation_matrix = zero_threshold(rotation_matrix, N_eps=5)

# Create Elasticity 4-Tensor
C = createCubicElasticityMatrix(c11, c12, c44)
# Display the non-zero components
print("Original Tensor in 6-space Tensor Notations:")
displayHookeLawMatrix(C)

# Create rotation matrix
R = rotation_matrix(np.pi/2)
print("Rotation Matrix: ")
print_cs(R)

# Rotate Elasticity Tensor
Cprime = transform_tensor(C, R)
print("Rotated Tensor in 6-space Tensor Notations:")
displayHookeLawMatrix(Cprime)

# Do Isotropic Case
c11_ios = c11
c12_iso = c12
c44_iso = 0.5*(c11 - c12)

print("Consider Isotropic Case: \n")
C_iso = createCubicElasticityMatrix(c11, c12, c44)
print("Isotropic Tensor in 6-space Tensor Notations:")
displayHookeLawMatrix(C_iso)

Cprime_iso = transform_tensor(C_iso, R)
print("Rotated Isotropic Tensor in 6-space Tensor Notations:")
displayHookeLawMatrix(Cprime_iso)

print("Difference between isotropic and rotated:\n ")
print_cs(C_iso - Cprime_iso)

print("Consider a random strain tensor")
E = np.random.uniform(size=[3,3])
E = 0.5*(E + E.T)
print_cs(E)

print("we obtain a stress tensor...")
S = np.tensordot(C, E, ((2,3), (0,1)))
print_cs(S)

print("Now rotating the stress and elasticity tensors...")

Cprime = transform_tensor(C, R)
# Eprime = transform_tensor(E, R)
# Sprime = transform_tensor(np.tensordot(Cprime, E, ((2,3), (0,1))), R.T)
Sprime = np.tensordot(Cprime, E, ((2,3), (0,1)))

print("we obtain a stress tensor...")
print_cs(Sprime)

print("With a diff...")
print_cs(Sprime-S)

# Cprime2 = brute_transform_tensor(C, R)
#
# M = displayHookeLawMatrix(Cprime2)


# v = np.array([1, 1, 1])
# s =1./np.sqrt(2)
# a_ij = np.array([[s, 0, -s],[0, 1, 0],[s, 0, s]])
# R = rotation_matrix(0, np.pi/4)
# print_cs(R.T)
# Cprime = transform_tensor(v, R.T)
# print_cs(Cprime)
# Cprime2 = transform_tensor(v, a_ij)
# print_cs(Cprime2)
# t_ij = np.array([[2,6,4], [0, 8, 0], [4, 2, 0]])
# Cprime3 = transform_tensor(t_ij, a_ij)
# print_cs(Cprime3)
# Cprime2 = transform_tensor(t_ij, R.T)
# print_cs(Cprime2)
