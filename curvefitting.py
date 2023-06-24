import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the array from the file
leftx = np.load('rightx.npy')
lefty = np.load('righty.npy')


def model_f(x, a, b, c):
    return a*(x-b)**2 + c

popt, pcov = curve_fit(model_f, leftx, lefty, p0=[3,2,-16])

points = (leftx,lefty)
print(points)

a_opt, b_opt, c_opt = popt
x_model = np.linspace(min(leftx), max(leftx), 1000)
print(min(leftx))
y_model = model_f(x_model, a_opt, b_opt, c_opt)
y_model = np.int32(y_model)
x_model = np.int32(x_model)
plt.scatter(leftx,lefty)
plt.plot(x_model,y_model, color='r')
plt.show()



# # Print the loaded array
# print(leftx.shape)
# print(lefty.shape)
# plt.scatter(leftx,lefty)
# plt.show()
