from elasticity_matrix import *
from array_utils import print_cs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

figDir = "../../figures/"
c11 = 1405.9  * 1.e9 # [Pa]
c12 = 1364.8  * 1.e9 # [Pa]
# NOTE: Taku's Values are given in Voight Notation, so there is an extra factor
#       of 2 in definition of c44: \hat{c44} = 2*c44 --> c44 = \hat{c44}/2
c44 =  397.9  * 1.e9# [Pa]
# c44 = c44_hat/2
rho =   12.98 * 1.e3 # [kg/m^3]

Pressure =  357.5 * 1.e9 # [Pa]
Temperature = 6000 # [K]

# Create Elasticity 4-Tensor
C = create_cubic_elasticity_tensor(c11, c12, c44)



# fig = plot_cubic_Pwavespeeds(c11, c12, c44, rho)
# plt.savefig(figDir+'analaytic_cubic_wavespeeds.pdf')
# plt.close()
#
# fig = plot_wavespeeds(*get_acoustic_Pwavespeeds(C, rho))
# plt.savefig(figDir+'../../numerical_cubic_wavespeeds.pdf')
# plt.close()

# # Display the non-zero components
# M = getHookeLawMatrix(C)
# print("Original Tensor in 6-space Tensor Notations:")
# print_cs(M)
#
# # Create rotation matrix
# R = rotation_matrix(np.pi/4, -55.*np.pi/180)
#
# #TODO: Add test to suite!
# # Rotate Elasticity Tensor
# Cprime = transform_tensor(C, R)
# fig = plot_wavespeeds(*get_acoustic_Pwavespeeds(Cprime, rho))
# plt.savefig(figDir+'rotated_cubic_wavespeeds.pdf')
# plt.close()


phi, theta, v_analytic = get_cubic_Pwavespeeds(c11, c12, c44, rho, 100)
phi, theta, v_numeric  = get_acoustic_Pwavespeeds(C, rho, 100)
phi, theta, v_all = get_eig_wavespeeds(C, rho)
v_eig = v_all[:, :, 2]
print(np.max(np.abs(v_analytic - v_numeric)/v_analytic))
print(np.max(np.abs(v_analytic - v_eig)/v_analytic))

f = plot_wavespeeds(phi, theta, v_analytic*1.e-3)
f.axes[0].set_title(r'$V_p$ Analytic')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'0_analaytic_cubic_Pwavespeeds.pdf')
plt.close(f)

f = plot_wavespeeds(phi, theta, v_numeric*1.e-3)
f.axes[0].set_title(r'$V_p$ from Christoffel Sum')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'1_numeric_brute_Pwavespeeds.pdf')
plt.close(f)

f = plot_wavespeeds(phi, theta, v_eig*1.e-3)
f.axes[0].set_title(r'$V_p$ from eigevalues of Christoffel Tensor')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'2_numeric_eig_Pwavespeeds.pdf')
plt.close(f)

f = plot_wavespeeds(phi, theta, (v_numeric-v_analytic)*1.e-3)
f.axes[0].set_title(r'Diff. $V_p$ Numeric Sum - Analytic')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'3_error_numeric_Pwavespeeds.pdf')
plt.close(f)

f = plot_wavespeeds(phi, theta, 100*(v_eig-v_analytic)/v_analytic)
f.axes[0].set_title(r'Diff. $V_p$ Eig. Christoffel - Analytic')
f.axes[1].set_ylabel(r'% $V_p$ Analytic')
f.savefig(figDir+'4_error_eigChristoffel_Pwavespeeds.pdf')
plt.close(f)
# plt.contourf(1.e-9*v_eig**2)

v_Smin= v_all[:, :, 0]
v_Smax= v_all[:, :, 1]
f = plot_wavespeeds(phi, theta, v_Smin*1.e-3)
f.axes[0].set_title(r'$V_{Smin}$ from Eigenvalues of Christoffel Tensor')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'5_Smin_eig_wavespeeds.pdf')
plt.close(f)

f = plot_wavespeeds(phi, theta, v_Smax*1.e-3)
f.axes[0].set_title(r'$V_{Smax}$ from Eigenvalues of Christoffel Tensor')
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'6_Smax_eig_wavespeeds.pdf')
plt.close(f)

num_chrystal = int(1.e5)
C_ave = np.zeros([3,3,3,3])
for ii in np.arange(num_chrystal):
    C_ave += transform_tensor(C, gen_rand_rot())/num_chrystal

phi, theta, v  = get_eig_wavespeeds(C_ave, rho, 50)
alpha = v[:, :, 2]
f = plot_wavespeeds(phi, theta, alpha*1.e-3)
f.axes[0].set_title(r'$V_p$ of {} randomly oriented cubic chrystals.'.format(num_chrystal))
f.axes[1].set_ylabel('km/s')
f.savefig(figDir+'7_randomlmy_oriented_Pwavespeeds.pdf')
plt.close(f)


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
