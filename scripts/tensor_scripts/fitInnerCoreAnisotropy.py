from elasticity_matrix import *
from array_utils import print_cs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import errno
import os

def safe_make(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def plot_wavespeeds(phi, theta, v, v_range=None):
    fig = plt.figure()
    if v_range is None:
        plt.contourf(phi, theta, v, 20)
    else:
        plt.contourf(phi, theta, v, 20, vmin=v_range[0], vmax=v_range[1])
    plt.colorbar()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\vartheta$')
    return fig

def plot_max(phi, theta, v):
    maxV = np.max(v)
    maxInd = np.unravel_index(np.argmax(v, axis=None), v.shape)
    phiMax = phi[maxInd]
    thetaMax = theta[maxInd]
    plt.plot(phiMax, thetaMax, 'xk', ms=5)
    plt.text(phiMax-12, thetaMax-10, "({:2.2f}, {:2.2f})".format(phiMax, thetaMax) )
    plt.text(phiMax-12, thetaMax+10, "{:2.2f} km/s".format(maxV) )

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


C = create_cubic_elasticity_tensor(c11, c12, c44)
phi, theta, V0 = get_eig_wavespeeds(C, rho)

curDir = figDir+'randomly_oriented/'
safe_make(curDir)
num_chrystal = int(1.e4)
R = randomEulerRotation()
C_ave = np.zeros([3,3,3,3])
for ii in np.arange(num_chrystal):
    C_ave += transform_tensor(C, R())/num_chrystal

f = R.plot_az_distribution()
f.savefig(curDir +'az_dist.pdf')
plt.close(f)

f = R.plot_axes_dist()
f.savefig(curDir +'axes_dist.pdf')
plt.close(f)


phi, theta, V_ave  = get_eig_wavespeeds(C_ave, rho, 50)
fname = ['Smin', 'Smax', 'P']
for ii in np.arange(3):
    v_min, v_max = np.min(V0[:,:,ii]*1.e-3), np.max(V0[:,:,ii]*1.e-3)
    v = V_ave[:, :, ii]
    fv = plt.figure()
    plt.pcolormesh(phi, theta, v*1.e-3, vmin=v_min, vmax=v_max,edgecolors='none')
    plt.colorbar()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\vartheta$')
    # fv = plot_wavespeeds(phi, theta, v*1.e-3, v_range=[v_min, v_max])
    fv.axes[0].set_title(r'$V_p$ of {} randomly oriented cubic crystals.'.format(num_chrystal))
    fv.axes[1].set_ylabel('km/s')
    fv.savefig(curDir + '{}_wavespeeds_N{}_pm.pdf'.format(fname[ii], num_chrystal))
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
