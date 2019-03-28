from elasticity_matrix import *
from params import *
from array_utils import print_cs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

d2r = np.pi / 180.
r2d = 180. / np.pi

res=20
# Inversion of A, C, F, L, N from Ishii 2002 & PREM constrains
PREM = tranverselyIsotropicCrystal(**prem_elastic_parms, rho=rho, wDir=wDir, ID='PREM', res=res)
FeSi = cubicCrystal(**FeSi_elastic_params, rho=rho)

N=10
count=0
residual = np.zeros([N,N])
kappa_list = np.linspace(1, 10, N)
theta_list = np.linspace(40, 60, N)
residual = np.zeros([len(theta_list),len(kappa_list)])
for ii, t in enumerate(theta_list):
    for jj, kappa in enumerate(kappa_list):
        R = rotation_matrix(0, t, 0)
        temp = transform_tensor(FeSi.Cijkl, R)
        VM = compositeElasticityTensor(temp, FeSi.rho, vonMisesRotate(0, kappa), \
                                            wDir=wDir, ID=str(count), res=res)
        VM.add_samples(5.e3)
        VM.ave_rot_axis(int_num=1000)
        residual[ii,jj] = np.sum((PREM.vel - VM.vel)**2)
        count += 1
        print("{} of {}".format(count, len(theta_list)*len(kappa_list)))
plt.pcolor(residual)
plt.colorbar()

col = np.argmin(np.min(residual, axis=1))
row = np.argmin(np.min(residual, axis=0))
kappa_best = kappa_list[row]
theta_best = theta_list[col]

# Best Composition 1
VM = compositeElasticityTensor(FeSi.Cijkl, FeSi.rho, vonMisesRotate(theta_best, kappa_best), \
                                            wDir=wDir, ID='BestComp1', res=res)
VM.add_samples(5.e3)
VM.ave_rot_axis(int_num=1000)
VM.plot_wavespeeds()
VM.plot_az_dist()
VM.plot_axes_dist()

diff = PREM.vel - VM.vel
for ii in np.arange(3):
    plot_wavespeeds(VM.phi, VM.theta, diff[:,:,ii]*1.e-3)
    plt.title('Diff. PREM & Best Compotision 1')
    plt.savefig(VM.fDir+"PREM_Diff_Comp_{}.pdf".format(VM.vel_names[ii]))
    plt.close()

for ii in np.arange(3):
    plot_wavespeeds(VM.phi, VM.theta, (diff[:,:,ii]/PREM.vel[:,:,ii])*100)
    plt.title('Diff. Percent PREM & Best Compotision 1')
    plt.savefig(VM.fDir+"PREM_Percent_Diff_Comp_{}.pdf".format(VM.vel_names[ii]))
    plt.close()


N=20
count=0
residual = np.zeros([N,N])
PHI, THETA = np.meshgrid(np.linspace(0, 360, N), np.linspace(0, 180, N))
for ii in np.arange(N):
    for jj in np.arange(N):
        p, t = PHI[ii,jj], THETA[ii, jj]
        R = rotation_matrix(p, t, 0)
        temp = transform_tensor(FeSi.Cijkl, R)
        # temp = compositeElasticityTensor(, FeSi.rho, zRotationDistribution(), res=N)
        temp = ave_rotate_z(temp)
        _, _, vel = get_eig_wavespeeds(temp, rho, res)
        residual[ii,jj] = np.sum((diff - vel)**2)
        count += 1
        print("{} of {}".format(count, int(N**2)))
plt.pcolor(PHI, THETA, residual)
plt.colorbar()
plt.savefig(VM.fDir+"Comp2_phi_theta_fit.pdf")
plt.close()

N=10
count=0
residual = np.zeros([N,N])
kappa_list = np.linspace(1, 10, N)
theta_list = np.linspace(0, 180, 3*N)
residual = np.zeros([len(theta_list),len(kappa_list)])
for ii, t in enumerate(theta_list):
    for jj, kappa in enumerate(kappa_list):
        R = rotation_matrix(0, t, 0)
        temp = transform_tensor(FeSi.Cijkl, R)
        VM = compositeElasticityTensor(temp, FeSi.rho, vonMisesRotate(0, kappa), \
                                            wDir=wDir, ID='BestComp2', res=res)
        VM.add_samples(5.e3)
        VM.ave_rot_axis(int_num=1000)
        residual[ii,jj] = np.sum((diff - VM.vel)**2)
        count += 1
        print("{} of {}".format(count, len(theta_list)*len(kappa_list)))
plt.pcolor(residual)
plt.colorbar()
plt.savefig("../../IC_compositions/BestComp1/figures/Comp2_kappa_theta_fit.pdf")
plt.close()

col = np.argmin(np.min(residual, axis=1))
row = np.argmin(np.min(residual, axis=0))
kappa_best2 = kappa_list[row]
theta_best2 = theta_list[col]


# Best Composition 1
VM = compositeElasticityTensor(temp, FeSi.rho, vonMisesRotate(theta_best2, kappa_best2), \
                                            wDir=wDir, ID='BestComp2', res=res)
VM.add_samples(5.e3)
VM.ave_rot_axis(int_num=1000)
VM.plot_wavespeeds()
VM.plot_az_dist()
VM.plot_axes_dist()

diff2 = diff - VM.vel
for ii in np.arange(3):
    plot_wavespeeds(VM.phi, VM.theta, diff2[:,:,ii]*1.e-3)
    plt.title('Diff. PREM & Best Compotision 1')
    plt.savefig(VM.fDir+"PREM_Diff_Comp_{}.pdf".format(VM.vel_names[ii]))
    plt.close()

###############
N=20
count=0
residual = np.zeros([N,N])
PHI, THETA = np.meshgrid(np.linspace(0, 360, N), np.linspace(0, 180, N))
for ii in np.arange(N):
    for jj in np.arange(N):
        p, t = PHI[ii,jj], THETA[ii, jj]
        R = rotation_matrix(p, t, 0)
        temp = transform_tensor(FeSi.Cijkl, R)
        # temp = compositeElasticityTensor(, FeSi.rho, zRotationDistribution(), res=N)
        temp = ave_rotate_z(temp)
        _, _, vel = get_eig_wavespeeds(temp, rho, res)
        residual[ii,jj] = np.sum((PREM.vel - vel)**2)
        count += 1
        print("{} of {}".format(count, int(N**2)))

plt.pcolor(PHI, THETA, residual)
plt.colorbar()
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$\vartheta$')

R = rotation_matrix(0, 85*d2r, 0)
temp = transform_tensor(FeSi.Cijkl, R)
temp = ave_rotate_z(temp)
phi, theta, vel = get_eig_wavespeeds(temp, rho, res)
diff = PREM.vel - vel
plt.contourf(phi, theta, diff[:,:,2]*1.e-3)
plt.colorbar()
plt.xlabel(r'$\varphi$')
plt.ylabel(r'$\vartheta$')
