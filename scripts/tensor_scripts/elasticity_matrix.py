import numpy as np
from functools import reduce
from array_utils import print_cs, zero_threshold
#import matplotlib as mpl
# from plot_utils import pgf_with_latex
# mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

def get_spherical_coordinates(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    return r, theta, phi

class randomRotation:
    def __init__(self):
        self.rot_log = []
        self.type = 'zyx'
        self._az_labels = {0: 'Z', 1: 'Y', 2: 'X'}
        self.prime_axes = [[],[],[]]


    def __call__(self):
        crot = self._gen_rand_rot_az()
        self.rot_log.append(crot)
        R = self._gen_rand_rot_matrix(crot)
        self._log_axes_distribution(R)
        return R

    def _gen_rand_rot_matrix(self, crot):
        return rotation_matrix(*crot)

    def _gen_rand_rot_az(self):
        return [gen_rand_az() for i in np.arange(3)]

    def _log_axes_distribution(self, R):
        for ii in np.arange(3):
            x_old = np.zeros([3,])
            x_old[ii] = 1
            x_new = transform_tensor(x_old, R)
            r, theta, phi = get_spherical_coordinates(*x_new)
            self.prime_axes[ii].append([theta, phi])

    def plot_az_distribution(self):
        dist = np.array(self.rot_log)
        f, axarr = plt.subplots(3, sharex=True, sharey=True)
        f.suptitle('Histogram of Rotation Angles')
        for ii in np.arange(3):
            axarr[ii].hist(dist[:,ii])
            axarr[ii].set_ylabel('{}-axis'.format(self._az_labels[ii]))
        return f

    def plot_axes_dist(self):
        f, axarr = plt.subplots(3, sharex=True, sharey=True)
        for ii in np.arange(3):
            dist = np.array(self.prime_axes[ii])
            x,y = dist[:,0],dist[:,1]
            axarr[ii].hist2d(x,y, 20)
            axarr[ii].set_xlabel(r'$\theta$')
            axarr[ii].set_ylabel(r'$\phi$')
        return f

class randomEulerRotation(randomRotation):
    def __init__(self):
        self.type = 'euler'
        self._az_labels = {0: r'$\alpha$', 1: r'$\beta$', 2: r'$\gamma$'}
        self.rot_log = []
        self.prime_axes = [[],[],[]]

    def _gen_rand_rot_az(self):
        alpha = np.random.uniform(0, 2*np.pi)
        # When sampling in the phi direction, we don't actually
        # want it to be uniform
        beta = np.arccos(1-2*np.random.uniform(0, 1))
        gamma = np.random.uniform(0, 2*np.pi)
        return [alpha, beta, gamma]

    def _gen_rand_rot_matrix(self, crot):
        return euler_rotation_matrix(*crot)

def gen_rand_az(az_range=[0, 2*np.pi]):
    return np.random.uniform(az_range[0], az_range[1])

def gen_rand_rot():
    return rotation_matrix(*[gen_rand_az() for i in np.arange(3)])

def gen_rand_mat(size=[]):
    return np.random.uniform(size=size)

def gen_rand_sym_mat(size=[]):
    temp = np.random.uniform(size=size)
    return temp + temp.T

def create_transversely_isotropic_matrix(A, C, F, L, N):
    '''Using Conventions of Tromp (1995) - see Equations (27)-(28).

       For more detailed explination see "Eigentensors of linear Anisotropic
       Materials" by Mehrabadi & Cowin (1989) - Equation 2.3
       C_1111 = C_2222 = A
       C_3333 = C
       C_1133 = C_2233 = F
       C_1313 = C_2323 = L
       C_1212 = N
       C_1122 = A - 2N '''
    # Divide by 2 because symmetrization will double the main diagonal
    # A trick to define only the top right half of the matrix
    V_ij = np.zeros([6,6])
    V_ij[0, 0] = V_ij[1, 1] = A/2. #C_1111 = C_2222 = A
    V_ij[2, 2] = C/2.              #C_3333 = C
    V_ij[3, 3] = V_ij[4, 4] = L/2. #C_2323 = C_1313 = L
    V_ij[5, 5] = N/2.              #C_1212 = N

    # Will symmetrize add end so only define half will pickup rest with transpose
    V_ij[0, 1] = A - 2*N           #C_1122 = A - 2N
    V_ij[0, 2] = V_ij[1, 2] = F    #C_1133 = C_2233 = F
    # symmetrize the hooke law matrix
    return V_ij + V_ij.T

def create_transversely_isotropic_tensor(A, C, F, L, N):
    '''Create 3x3x3x3 Elasticity Tensor with transverse isotropy'''
    V_ij = create_transversely_isotropic_matrix(A, C, F, L, N)
    return create_Cijkl_from_hooke_law_matrix(V_ij)

def create_Cijkl_from_hooke_law_matrix(V_ij):
    """Turn a 6x6 matrix into a 3x3x3x3 tensor according to Voigt notation."""
    # Definition of Voigt notation
    VOIGT = {0: 0, 11: 1, 22: 2, 12: 3, 21: 3, 2: 4, 20: 4, 1: 5, 10: 5}
    C_ijkl = [[[[V_ij[VOIGT[10*i+j]][VOIGT[10*k+l]]
                 for i in range(3)] for j in range(3)]
                 for k in range(3)] for l in range(3)]
    return np.array(C_ijkl)

def create_cubic_hooke_law_matrix(c11, c12, c44):
    V_ij = np.zeros([6,6])
    V_ij[0, 0] = V_ij[1, 1] = V_ij[2, 2] = c11 #C_1111 = C_2222 = C_3333 = c11
    V_ij[3, 3] = V_ij[4, 4] = V_ij[5, 5] = c44 #C_2323 = C_1313 = C_1212 = c44
    V_ij[0, 1] = V_ij[0, 2] = V_ij[1, 2] = c12 #C_1313 = C_2323 = C_1212 = c12
    V_ij[1, 0] = V_ij[2, 0] = V_ij[2, 1] = c12 #C_1313 = C_2323 = C_1212 = c12
    return V_ij

def create_cubic_elasticity_tensor(c11, c12, c44):
    '''Create 3x3x3x3 Elasticity Tensor with cubic symmetry'''
    V_ij = create_cubic_hooke_law_matrix(c11, c12, c44)
    return create_Cijkl_from_hooke_law_matrix(V_ij)

def brute_create_cubic_elasticity_tensor(c11, c12, c44):
    '''Using Conventions detailed in "Eigentensors of linear Anisotropic
        Materials" by Mehrabadi & Cowin (1989).

        BE SURE TO CHECK PG 134 of NYE 4->(23,32), 5->(31,13), 6->(12,21)
    '''

    C = np.zeros([3,3,3,3])

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

def get_hooke_law_matrix(C_ijkl):
    """Turn a 3x3x3x3 tensor to a 6x6 matrix according to Voigt notation."""
    C_ij = np.zeros((6,6))

    # Divide by 2 because symmetrization will double the main diagonal
    C_ij[0,0] = 0.5*C_ijkl[0][0][0][0]
    C_ij[1,1] = 0.5*C_ijkl[1][1][1][1]
    C_ij[2,2] = 0.5*C_ijkl[2][2][2][2]
    C_ij[3,3] = 0.5*C_ijkl[1][2][1][2]
    C_ij[4,4] = 0.5*C_ijkl[0][2][0][2]
    C_ij[5,5] = 0.5*C_ijkl[0][1][0][1]

    C_ij[0,1] = C_ijkl[0][0][1][1]
    C_ij[0,2] = C_ijkl[0][0][2][2]
    C_ij[0,3] = C_ijkl[0][0][1][2]
    C_ij[0,4] = C_ijkl[0][0][0][2]
    C_ij[0,5] = C_ijkl[0][0][0][1]

    C_ij[1,2] = C_ijkl[1][1][2][2]
    C_ij[1,3] = C_ijkl[1][1][1][2]
    C_ij[1,4] = C_ijkl[1][1][0][2]
    C_ij[1,5] = C_ijkl[1][1][0][1]

    C_ij[2,3] = C_ijkl[2][2][1][2]
    C_ij[2,4] = C_ijkl[2][2][0][2]
    C_ij[2,5] = C_ijkl[2][2][0][1]

    C_ij[3,4] = C_ijkl[1][2][0][2]
    C_ij[3,5] = C_ijkl[1][2][0][1]

    C_ij[4,5] = C_ijkl[0][2][0][1]
    return C_ij + C_ij.T

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

@zero_threshold
def euler_rotation_matrix(alpha, beta, gamma):
    assert 0 <= alpha <= 2*np.pi, "Alpha must be between 0 and 2pi"
    assert 0 <= beta <= np.pi, "Beta must be between 0 and pi"
    assert 0 <= gamma <= 2*np.pi, "Gamma must be between 0 and 2pi"
    A = rotation_matrix(z=alpha)
    B = rotation_matrix(y=beta)
    C = rotation_matrix(z=gamma)
    return np.dot(C, np.dot(B,A))

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

def get_direction_vector(phi, theta):
    """ q = cos(phi)sin(theta)x_1 + sin(phi)sin(theta)x_2 + cos(theta)x_3"""
    q = np.zeros(shape=[3,])
    q[0] = np.cos(phi) * np.sin(theta)
    q[1] = np.sin(phi) * np.sin(theta)
    q[2] = np.cos(theta)
    return q

def get_christoffel_tensor(C_ijkl, q):
    # M_ij = \q_n C_{inmj} \q_m
    M_ij  = np.tensordot(q, np.tensordot(q, C_ijkl, (0,1)), (0,2))
    return M_ij

def get_wavespeed(C_ijkl, rho, q, u):
    """ \rho v^2 = q_i u_j c_{ijkl} q_k u_l
    """
    # Create Christoffel Matrix
    M_ij  = get_christoffel_tensor(C_ijkl, q)
    rhov2 = np.dot(u, np.dot(u, M_ij))
    return np.sqrt(rhov2/rho)

def get_eig_wavespeeds(C_ijkl, rho, N=100):
    #TODO: Vectorize this!!
    PHI, THETA = np.meshgrid(np.linspace(0, 360, N), np.linspace(0, 180, N))
    V = np.zeros([N, N, 3])
    # Need vectorize this
    for ii in np.arange(N):
        for jj in np.arange(N):
            phi   = PHI[ii, jj]
            theta = THETA[ii, jj]
            q = get_direction_vector(phi*np.pi/180, theta*np.pi/180)
            M_ij  = get_christoffel_tensor(C_ijkl, q)
            cons = np.max(M_ij)
            U, W = np.linalg.eig(M_ij/cons)
            # Theor. e-values are 1 +/- 1e-9
            V[ii, jj, :] = np.sort(np.sqrt(cons*U/rho))
    return PHI, THETA, V

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

def get_bulk(C):
    M = get_hooke_law_matrix(C)
    return (M[0][0]+M[1][1]+M[2][2]+2*(M[0][1]+M[0][2]+M[1][2]))/9

def get_shear(C):
    M = get_hooke_law_matrix(C)
    return ((M[0][0]+M[1][1]+M[2][2]) - (M[0][1]+M[0][2]+M[1][2]) \
            + 3*(M[3][3]+M[4][4]+M[5][5]))/15

# Create test for this by checking on ISOtropic matrices, Cubic which we
# hvae expressions for the isotropic part and rotating and averaging a generic matrix
def get_iso_velocites(C, rho):
    bulk = get_bulk(C)
    shear = get_shear(C)
    P  = np.sqrt((bulk + 4.0*shear/3)/rho)
    S  = np.sqrt(shear/rho)
    return P, S
    
