#Work out the Mean (the simple average of the numbers)
#Then for each number: subtract the Mean and square the result.
#Then work out the mean of those squared differences.
#Take the square root of that and we are done!
import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import time
import math

def calculate_Mean(gaussian, length):
    total_value = 0
    for i in range(0, length):
        total_value = total_value + gaussian[i]
    total_value = total_value / length
    return total_value

def calculate_Std_dev(gaussian, length, mean):
    total_error = 0
    for i in range(0, length):
        error = math.pow((mean - gaussian[i]), 2)
        total_error = total_error + error
    total_error = total_error / (length - 1)
    total_error = math.sqrt(total_error)
    return  total_error


Gauss1 = np.random.normal(0,5, 25000)
new_mean = calculate_Mean(Gauss1, 25000)
new_std_dev = calculate_Std_dev(Gauss1, 25000, new_mean)

print(new_mean)
print(new_std_dev)

time.sleep(3)