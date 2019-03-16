from elasticity_matrix import *
from params import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

# Isotropic Distribution
R = randomEulerRotation()
FeSi = cubicCrystal(**FeSi_elastic_params, rho=rho)
Cijkl = compositeElasticityTensor(FeSi.Cijkl, FeSi.rho, R, wDir = wDir)
Cijkl.add_samples(1.e3)
Cijkl.plot_wavespeeds()
Cijkl.plot_az_dist()
Cijkl.plot_axes_dist()

# Zonal Distribution
Z = zRotationDistribution()
Zijkl = compositeElasticityTensor(FeSi.Cijkl, FeSi.rho, Z, wDir=wDir, ID='Zrot')
Zijkl.add_samples(1.e3)
Zijkl.plot_wavespeeds()
Zijkl.plot_az_dist()
Zijkl.plot_axes_dist()

PREM = tranverselyIsotropicCrystal(**prem_elastic_parms, rho=rho,wDir=wDir, ID='PREM')
PREM.plot_wavespeeds()
