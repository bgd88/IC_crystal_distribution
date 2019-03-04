import numpy as np
from functools import reduce
import colorama

def createCubicElasticityMatrix(c11, c12, c44):
    '''Using Conventions detailed in "Eigentensors of linear Anisotropic
        Materials" by Mehrabadi & Cowin (1989).
    '''

    C = np.zeros([3,3,3,3]);

    C[0,0,0,0] = c11;
    C[1,1,0,0] = c12;
    C[2,2,0,0] = c12;

    C[0,0,1,1] = c12;
    C[1,1,1,1] = c11;
    C[2,2,1,1] = c12;

    C[0,0,2,2] = c12;
    C[1,1,2,2] = c12;
    C[2,2,2,2] = c11;
    # Factor of 2 from Voight Notation to 4 Tensor
    C[1,2,1,2] = c44/2;
    C[0,2,0,2] = c44/2;
    C[0,1,0,1] = c44/2;
    return C


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

def rotT(T, g):
    Tprime = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    gg = g[ii,i]*g[jj,j]*g[kk,k]*g[ll,l]
                                    Tprime[i,j,k,l] = Tprime[i,j,k,l] + \
                                         gg*T[ii,jj,kk,ll]
    return Tprime

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

    with np.printoptions(linewidth=100, formatter={'float': color_sign}):
        print(M)
        print("\n")
    return M

def color_sign(x):
    if x > 0:
        c = colorama.Fore.GREEN
        x = '{:2.2E}'.format(x)
    elif x < 0:
        c = colorama.Fore.RED
        x = '{:2.2E}'.format(x)
    else:
        c = colorama.Fore.WHITE
        x = '{:08f}'.format(x)
    return f'{c}{x}'
