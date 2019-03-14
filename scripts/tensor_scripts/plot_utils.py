import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def plot_cubic_Pwavespeeds(c11, c12, c44, rho):
    """ C_{ijkl} = \lambda \delta_{ij}\delta_{kl} +
                    2\mu*(\delta_{il}\delta_{jk} + \delta_{ik}\delta_{jl})
        c_{11} = \lambda + 2\mu + \eta
        c_{12} = \lambda
        c_{44} = \mu
    """
    phi, theta, v = get_cubic_Pwavespeeds(c11, c12, c44, rho)
    fig = plot_wavespeeds(phi, theta, v*1.e-3)
    return
