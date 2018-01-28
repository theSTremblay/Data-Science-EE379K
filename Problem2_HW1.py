import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import time
def gauss(x_array, mu, sigma):
    pdf = scipy.stats.norm.pdf(x_array, loc=mu, scale=sigma)
    return pdf
def Problem2():
    x_bernoulli= {-1,1}

    z_array = []

    for j in range(0,1000):

        bernoulli_1 = (scipy.stats.bernoulli.rvs(.5, size=1000) *2) -1

        # counter = 0
        # for i in bernoulli_1:
        #     if i == 0:
        #         bernoulli_1[counter] = -1
        #     counter = counter + 1
        z_array.append(bernoulli_1.mean())

    #binomial_value = scipy.stats.bernoulli.rvs(x_bernoulli, size=1000)
    nd_z_array = np.asarray(z_array)

    plt.hist(nd_z_array, normed=True, bins=100)
    plt.ylabel('Probability scaled by 1000')
    plt.show()

    bernoulli_2 = scipy.stats.bernoulli.rvs(.5, size=5)

    counter = 0
    for i in bernoulli_2:
        if i == 0:
            bernoulli_2[counter] = -1
        counter = counter + 1

    plt.hist(bernoulli_2, normed=True, bins=100)
    plt.ylabel('Probability_medium_check')
    plt.show()

    bernoulli_3 = scipy.stats.bernoulli.rvs(.5, size=250)

    counter = 0
    for i in bernoulli_3:
        if i == 0:
            bernoulli_3[counter] = -1
        counter = counter + 1

    plt.hist(bernoulli_3, normed=True, bins=100)
    plt.ylabel('Probability_medium_check')

    plt.show()

Problem2()

time.sleep(3)