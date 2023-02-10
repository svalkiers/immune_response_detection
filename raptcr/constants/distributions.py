import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu, sigma):
    """
    Mathematical formulation of the normal distribution.

    Parameters
    ----------
    x
        x values of the distribution.
    mu
        Mean of the distribution.
    sigma
        Standard deviation around mu.

    """
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def exponential(x, beta):
    """
    Mathematical formulation of an exponential function.

    Parameters
    ----------
    x
        x values of the distribution.
    beta
        Scale parameter.
    """
    return (1/beta)*(np.exp(-x/beta))


def gaussian_vector(nbins=64, mu=1, sigma=.1, n=10000):
    """
    Create a vector of ordered, binned probability densities from the normal distribution.

    Parameters
    ----------
    nbins
        Number of data points.
    mu
        Mean of the distribution.
    sigma
        Standard deviation around mu.
    n
        Number of samples to draw from the normal distribution.
    """
    s = np.random.normal(mu, sigma, n)
    count, bins, ignored = plt.hist(s, nbins-1, density=True)
    plt.close()
    return np.array(normal(x=bins, mu=mu, sigma=sigma))

def negexp_vector(nbins, beta=1, n=10000):
    """
    Create a vector of ordered, binned probability densities from a parametrized negative exponential distribution.

    Parameters
    ----------
    nbins
        Number of data points.
    beta
        Scale parameter.
    n
        Number of samples to draw from the exponential distribution.
    """
    s = np.random.exponential(beta, n)
    count, bins, ignored = plt.hist(s, nbins-1, density=True)
    plt.close()
    return np.array(exponential(bins, beta)) 