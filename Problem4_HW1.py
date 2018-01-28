import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import time
import math

def calculate_Mean_multi_Dem(gaussian, length, dimension):
    total_array = []
    for j in range(0 , dimension):
        total_value = 0
        for i in range(0, length):
            total_value = total_value + gaussian[i][j]
        total_value = total_value / length
        total_array.append(total_value)
    return(total_array)
def calculate_variance(gaussian, length, mean, dimension):
    total_error = 0
    for i in range(0, length):
        error = math.pow((mean[dimension] - gaussian[i][dimension]), 2)
        total_error = total_error + error
    total_error = total_error / length
    #total_error = math.sqrt(total_error)
    return  total_error



def calculate_Std_dev_Multi_Dem(gaussian, length, mean, dimension):
    total_array = []
    variance_x = calculate_variance(gaussian, 10000, mean, 0)
    variance_y = calculate_variance(gaussian, 10000, mean, 1)
    # covariance
    #for j in range(0, dimension):
    total_errorx = 0
    total_errory = 0
    total_error = 0
    for i in range(0, length):
        error_x = (mean[0] - gaussian[i][0])
        error_y = (mean[1] - gaussian[i][1])

        #total_errorx = total_errorx + error_x
        #total_errory = total_errory + error_y
        total_error = total_error + (error_x *error_y)
    total_error = total_error / (length - 1)
    #total_errorx = math.sqrt(total_errorx)
    #total_errory = total_errory / length
    #total_errory = math.sqrt(total_errory)


    total_array = [[variance_x, total_error], [variance_y, total_error]]
    return (total_array)




mean = (-5, 5)
cov = [[20, .8], [.8, 30]]

multi_Gauss = np.random.multivariate_normal(mean , cov , 10000)

new_mean = calculate_Mean_multi_Dem(multi_Gauss, 10000, 2)
new_STD_DEV = calculate_Std_dev_Multi_Dem(multi_Gauss, 10000, new_mean, 2)

print(new_mean)
print(new_STD_DEV)



print("0")