from elasticity_matrix import *
from params import *
from array_utils import print_cs
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Isotropic Distribution
R = randomEulerRotation()
sc_Cijkl = create_cubic_elasticity_tensor(**FeSi_elastic_params)
Cijkl = compositeElasticityTensor(sc_Cijkl, rho, R, wDir = wDir)
Cijkl.add_samples(1.e4)
Cijkl.plot_wavespeeds()
Cijkl.plot_az_dist()
Cijkl.plot_axes_dist()

# Zonal Distribution
Z = zRotationDistribution()
Zijkl = compositeElasticityTensor(sc_Cijkl, rho, Z, wDir=wDir, ID='Zrot')
Zijkl.add_samples(1.e3)
Zijkl.plot_wavespeeds()
Zijkl.plot_az_dist()
Zijkl.plot_axes_dist()

# Inversion of A, C, F, L, N from Ishii 2002 & PREM constrains
Pijkl = create_transversely_isotropic_tensor(**prem_elastic_parms)
phi, theta, V = get_eig_wavespeeds(Pijkl, rho)
