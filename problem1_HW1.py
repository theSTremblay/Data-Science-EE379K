import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
def gauss(x_array, mu, sigma):
    pdf = scipy.stats.norm.pdf(x_array, loc=mu, scale=sigma)
    return pdf


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#for mu, sig in [(-10, 5), (10, 5)]:
 #   plt.plot(gaussian(np.linspace(-3, 3, 120), mu, sig))


def Problem_1a():
    #xarray1 = np.empty(1000, dtype=object)
    #Gauss1 = gauss(xarray1, (-10), 5)

    #xarray2 = np.empty(1000, dtype=object)
    #Gauss2 = gauss(2, 10, 5)

    Gauss1 = np.random.normal(10,5, 1000)
    Gauss2 = np.random.normal(-10, 5, 1000)

    TotalGauss = Gauss1 + Gauss2

    plt.hist(TotalGauss, normed=True, bins=30)
    plt.ylabel('Probability')
# Part 2
    std_dev = np.std(TotalGauss)
    mean = np.mean(TotalGauss)

    print(std_dev)
    print(mean)

Problem_1a()


