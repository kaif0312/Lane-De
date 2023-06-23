import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import cv2 as cv2
#import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
# y = [0,1,3,4,3,5,7,5,2,3,4,8,9,8,7]
# n = len(y)
# x = range(0, n)

# plt.plot(x, y, 'ro', label="original")
# plt.plot(x, y, 'b', label="linear interpolation")
# plt.title("Target data")
# plt.legend(loc='best', fancybox=True, shadow=True)
# plt.grid()
# plt.show()

# tck = interpolate.splrep(x, y, s=0, k=3)
# x_new = np.linspace(min(x), max(x), 100)
# y_fit = interpolate.BSpline(*tck)(x_new)

# plt.title("BSpline curve fitting")
# plt.plot(x, y, 'ro', label="original")
# plt.plot(x_new, y_fit, '-c', label="B-spline")
# plt.legend(loc='best', fancybox=True, shadow=True)
# plt.grid()
# plt.show() 

def perspective_warp(img,
                     dst_size=(1920,1080),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

img = cv2.imread('data/image0219.jpg')
print(img.shape)
dst = perspective_warp(img)

#cv2.imshow('Image', dst)
output_path = 'transform.jpg'

cv2.imwrite(output_path, dst
)
# Wait for a key press to close the window
#cv2.waitKey(10000)

# Close all open windows
#cv2.destroyAllWindows()