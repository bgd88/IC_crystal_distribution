import numpy as np
from functools import reduce
from array_utils import print_cs, zero_threshold

def createCubicElasticityMatrix(c11, c12, c44):
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

    C[1,2,1,2] = c44
    C[1,2,2,1] = c44
    C[2,1,2,1] = c44
    C[2,1,1,2] = c44

    C[0,2,0,2] = c44
    C[2,0,0,2] = c44
    C[0,2,2,0] = c44
    C[2,0,2,0] = c44

    C[1,0,1,0] = c44
    C[0,1,0,1] = c44
    C[0,1,1,0] = c44
    C[1,0,0,1] = c44

    return C

def createIsotropicElasticityMatrix(lam, mu):
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
    return createCubicElasticityMatrix(c11, c12, c44)

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

def displayHookeLawMatrix(C):

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

    print_cs(M)
    return M
