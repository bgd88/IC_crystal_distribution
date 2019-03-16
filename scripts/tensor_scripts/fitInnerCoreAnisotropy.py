from elasticity_matrix import *
from params import *
from array_utils import print_cs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

res=50
# Inversion of A, C, F, L, N from Ishii 2002 & PREM constrains
PREM = tranverselyIsotropicCrystal(**prem_elastic_parms, rho=rho,wDir=wDir, ID='PREM', res=res)
FeSi = cubicCrystal(**FeSi_elastic_params, rho=rho)

N=10
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

plt.pcolormesh(PHI, THETA, residual)
