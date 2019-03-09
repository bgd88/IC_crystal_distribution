import numpy as np
from functools import reduce
from array_utils import print_cs, zero_threshold
#import matplotlib as mpl
# from plot_utils import pgf_with_latex
# mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

def create_cubic_elasticity_tensor(c11, c12, c44):
    '''Using Conventions detailed in "Eigentensors of linear Anisotropic
        Materials" by Mehrabadi & Cowin (1989).

        BE SURE TO CHECK PG 134 of NYE 4->(23,32), 5->(31,13), 6->(12,21)
    '''

    C = np.zeros([3,3,3,3]);

    C[0,0,0,0] = c11
    C[1,1,0,0] = c12
    C[2,2,0,0] = c12

    C[0,0,1,1] = c12
    C[1,1,1,1] = c11
    C[2,2,1,1] = c12

    C[0,0,2,2] = c12
    C[1,1,2,2] = c12
    C[2,2,2,2] = c11

    C[1,2,1,2] = C[1,2,2,1] = C[2,1,2,1] = C[2,1,1,2] = c44
    C[0,2,0,2] = C[2,0,0,2] = C[0,2,2,0] = C[2,0,2,0] = c44
    C[1,0,1,0] = C[0,1,0,1] = C[0,1,1,0] = C[1,0,0,1] = c44
    return C

def create_isotropic_elasticity_tensor(lam, mu):
    """ Define using Lamé parameters (\lambda, \mu) such that:
        C_{ijkl} = \lambda \delta_{ij}\delta_{kl} +
            2\mu*(\delta_{il}\delta_{jk} + \delta_{ik}\delta_{jl})
        where \delta_{ij} is the Kronecker delta. So that the stress-strain
        relationship becomes:
        \tau_{ij} = \lambda\delta_{ij}e_{kk} + 2\mue_{ij}

        In terms of the 6-space vector notation:
        \lambda = c12
        \mu     = (c11 - c12)/2 """
    c12 = lam
    c11 = lam + 2*mu
    c44 = mu # Need the two since we expect the Voight C11, C22
    return create_cubic_elasticity_tensor(c11, c12, c44)

@zero_threshold
def rotation_matrix(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    If more than one rotation is given then they are applied
    first about the z, then y, then x

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Notes
    -----
    see: https://en.wikipedia.org/wiki/Euler_angles

    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def transform_tensor(T, L):
    ''' Transform an n-th order Tensor using a matrix.

        Parameters
        ----------
        T : numpy ndarray
            n-th order tensor to be transformed
        R : numpy ndarray
            matrix tranform

        Returns
        -------
        Tprime : numpy ndarray
                 transformed tensor
    '''
    # Determine Tensor Order
    Nord = len(T.shape)
    rn = range(Nord)

    # 0 is for outer product i.e. R_ij*R_lm =RR_ijlm
    Lc = reduce(lambda x,y:np.tensordot(x,y,0),[L for ii in rn])
    # Sum every component of tensor with every other of transformation
    # i.e. T'_ijk = R_ia*R_jb*R_kc... * T_abc...
    Tprime = np.tensordot(Lc, T, (tuple(2*t + 1 for t in rn), tuple(rn)))
    return Tprime

def brute_transform_tensor(T,tmx):
    #
    # FUNCTION
    # otr = transform(itr,tmx)
    #
    # DESCRIPTION
    # transform 3D-tensor (Euclidean or Cartesion tensor) of any order (>0) to another coordinate system
    #
    # PARAMETERS
    # otr = output tensor, after transformation; has the same dimensions as the input tensor
    # itr = input tensor, before transformation; should be a 3-element vector, a 3x3 matrix, or a 3x3x3x... multidimensional array, each dimension containing 3 elements
    # tmx = transformation matrix, 3x3 matrix that contains the direction cosines between the old and the new coordinate system

    ne = T.size                     # number of tensor elements
    init_shape = T.shape            # initial tensor shape
    nd = len(init_shape)             # number of tensor dimensions, i.e. order of tensor
    itr = T.flatten(order='C')      # flatten array
    otr = np.zeros(itr.shape)        # create output tensor

    iie = np.zeros([nd,1])          # initialise vector with indices of input tensor element
    ioe = np.zeros([nd,1])          # initialise vector with indices of output tensor element
    cne = (np.cumprod(3*np.ones([nd,1]))/3)  # vector with cumulative number of elements for each dimension (divided by three)

    for oe in np.arange(ne):                  # loop over all output elements
        tmp = 0
        ioe = np.mod(np.floor((oe)/cne),3)     # calculate indices of current output tensor element
        for ie in np.arange(ne):               # loop over all input elements
            pmx = 1                             # initialise product of transformation matrices
            iie = np.mod(np.floor((ie)/cne),3)       # calculate indices of current input tensor element
            for id in np.arange(nd):   # loop over all dimensions
                pmx = pmx * tmx[ int(ioe[id]), int(iie[id]) ]  # create product of transformation matrices
            otr[oe]  = otr[oe] + pmx * itr[ie]       # add product of transformation matrices and input tensor element to output tensor element
    return otr.reshape(init_shape, order='C')

def getHookeLawMatrix(C):

    M = np.zeros([6, 6])

    M[0,0] = C[0,0,0,0]
    M[1,0] = C[1,1,0,0]
    M[2,0] = C[2,2,0,0]
    M[3,0] = C[1,2,0,0]
    M[4,0] = C[0,2,0,0]
    M[5,0] = C[0,1,0,0]

    M[0,1] = C[0,0,1,1]
    M[1,1] = C[1,1,1,1]
    M[2,1] = C[2,2,1,1]
    M[3,1] = C[1,2,1,1]
    M[4,1] = C[0,2,1,1]
    M[5,1] = C[0,1,1,1]

    M[0,2] = C[0,0,2,2]
    M[1,2] = C[1,1,2,2]
    M[2,2] = C[2,2,2,2]
    M[3,2] = C[1,2,2,2]
    M[4,2] = C[0,2,2,2]
    M[5,2] = C[0,1,2,2]

    M[0,3] = C[0,0,1,2]
    M[1,3] = C[1,1,1,2]
    M[2,3] = C[2,2,1,2]
    M[3,3] = C[1,2,1,2]
    M[4,3] = C[0,2,1,2]
    M[5,3] = C[0,1,1,2]

    M[0,4] = C[0,0,0,2]
    M[1,4] = C[1,1,0,2]
    M[2,4] = C[2,2,0,2]
    M[3,4] = C[1,2,0,2]
    M[4,4] = C[0,2,0,2]
    M[5,4] = C[0,2,0,1]
    M[5,5] = C[0,1,0,1]
    return M

def get_direction_vector(phi, theta):
    """ q = cos(phi)sin(theta)x_1 + sin(phi)sin(theta)x_2 + cos(theta)x_3"""
    q = np.zeros(shape=[3,])
    q[0] = np.cos(phi) * np.sin(theta)
    q[1] = np.sin(phi) * np.sin(theta)
    q[2] = np.cos(theta)
    return q

def get_wavespeed(C_ijkl, rho, q, u):
    """ \rho v^2 = q_i u_j c_{ijkl} q_k u_l
    """
    # Create Christoffel Matrix
    # M_ij = \q_n C_{inmj} \q_m
    M_ij  = np.tensordot(q, np.tensordot(q, C_ijkl, (0,1)), (0,2))
    # M_ij = np.dot(q, np.dot(q, C_ijkl))
    rhov2 = np.dot(u, np.dot(u, M_ij))
    return np.sqrt(rhov2/rho)

def get_acoustic_Pwavespeeds(C_ijkl, rho, N=100):
    #TODO: Vectorize this!!
    PHI, THETA = np.meshgrid(np.linspace(0, 360, N), np.linspace(0, 180, N))
    V = np.zeros_like(THETA)
    # Need vectorize this
    for ii in np.arange(N):
        for jj in np.arange(N):
            phi   = PHI[ii, jj]
            theta = THETA[ii, jj]
            q = get_direction_vector(phi*np.pi/180, theta*np.pi/180)
            V[ii, jj] = get_wavespeed(C_ijkl, rho, q, q)
    return PHI, THETA, V

def plot_wavespeeds(phi, theta, v):
    fig = plt.figure()
    plt.contourf(phi, theta, v, 20)
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

def get_cubic_Pwavespeeds(c11, c12, c44, rho, N=100):
    """ Returns Longitudinal Cubic Wavespeeds as a function of two angles
    phi is the angle the vector makes with the x1 direction in the x1-x2
    plane and theta is the angle between the vector and x3

    Following Miaki's note on Acoustic wave speeds and Zheng & Spencer 1993:
    c11 = lambda + 2*mu + eta
    c12 = lambda
    c13 = mu"""
    mu = c44;
    lam = c12
    eta = c11 - lam - 2*mu

    # Generate 0..90 degree grids
    phi, theta = np.meshgrid(np.linspace(0, 360, N), np.linspace(0, 180, N))

    cT4 = np.cos(theta* np.pi / 180. )**4
    cP4 = np.cos(phi* np.pi / 180. )**4
    sP4 = np.sin(phi* np.pi / 180. )**4
    sT4 = np.sin(theta* np.pi / 180. )**4
    rhov2 = lam + 2*mu + eta*(cP4*sT4 + sP4*sT4 + cT4)

    v = np.sqrt(rhov2/rho)
    return phi, theta, v

def plot_cubic_Pwavespeeds(c11, c12, c44, rho):
    """ C_{ijkl} = \lambda \delta_{ij}\delta_{kl} +
                    2\mu*(\delta_{il}\delta_{jk} + \delta_{ik}\delta_{jl})
        c_{11} = \lambda + 2\mu + \eta
        c_{12} = \lambda
        c_{44} = \mu
    """
    phi, theta, v = get_cubic_Pwavespeeds(c11, c12, c44, rho)
    fig = plot_wavespeeds(phi, theta, v*1.e-3)
    return fig
