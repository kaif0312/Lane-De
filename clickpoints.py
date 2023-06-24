import cv2 as cv2
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import matplotlib as plt
estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42)),
    ("RANSAC", RANSACRegressor(random_state=42)),
    ("HuberRegressor", HuberRegressor()),
]

# Load the image
image = cv2.imread('data/image0120.jpg')
image_width = image.shape[1]
image_height = image.shape[0]
print(image.shape)
# cv2.imshow('show',image)
image = cv2.GaussianBlur(image, (3, 3), 0)
image = cv2.GaussianBlur(image, (3, 3), 0)
image = cv2.GaussianBlur(image, (3, 3), 0)
# Create a list to store the selected points
points = []
# Mouse click event handler
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point to the list
        points.append((x, y))
        # Draw a circle at the clicked point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # Display the image with selected points
        cv2.imshow('Image', image)

# Create a window and bind the mouse callback function
# cv2.namedWindow('Image')
# cv2.setMouseCallback('Image', mouse_callback)

# # Display the image and wait for points selection
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# Check if four points have been selected
if len(points) == 0:
    # Define the four source points (selected points)
    points_defined = [(520,265),(1372,265),(1844,745),(74,745)] # Defined points that would give somewhat perspective transform
    src_points = np.float32(points_defined) 

    # Define the four destination points for the perspective transform
    # This will define a rectangle in the transformed image
    # Adjust the destination points based on your desired perspective
    dst_points = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]) # points will be transformed to whole image
    # dst_points = np.float32([(74,265),(1844,265),(1844,745),(74,745)])
    #dst_points = np.float32([(points[3][0],points[0][1]),(points[2][0],points[0][1]),points[2],points[3]])
    # Compute the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_perspective_matrix = np.linalg.inv(perspective_matrix)
    # rotation_matrix = np.array([[1055.55615, 0.00000, 948.64667],
    #                             [0.00000, 1055.55615, 572.65509],
    #                             [0.00000, 0.00000, 1.00000]])

    # Incorporate rotation into the perspective transformation matrix
    # perspective_matrix = np.dot(perspective_matrix, rotation_matrix)

    # Perform the perspective transform
    warped_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))
    cv2.imshow('warped image',warped_image)
    unwarped_image = cv2.warpPerspective(warped_image, inverse_perspective_matrix, (image.shape[1], image.shape[0]))
    def sobel_filtering(warped_img):
      hls_image = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HLS)
      h, l, s = cv2.split(hls_image)
      sobel_x_s = cv2.Sobel(s, cv2.CV_64F, 1, 0)
      sobel_x_l = cv2.Sobel(l, cv2.CV_64F, 1, 0)
      abs_sobel_x_s = np.absolute(sobel_x_s)
      abs_sobel_x_l = np.absolute(sobel_x_l)
      s_threshold = 50  # Adjust the threshold values as needed
      l_threshold = 50
      s_binary = np.zeros_like(s, dtype=np.uint8)
      s_binary[(abs_sobel_x_s >= s_threshold)] = 1
      l_binary = np.zeros_like(l, dtype=np.uint8)
      l_binary[(abs_sobel_x_l >= l_threshold)] = 1

      # Combine binary images for saturation and lightness channels
      combined_binary = cv2.bitwise_or(s_binary, l_binary)
      binary_image = np.array(combined_binary, dtype=np.uint8) * 255
      return binary_image
    binary_image = sobel_filtering(image)
    cv2.imshow('edge image',binary_image)
    binary_image = cv2.warpPerspective(binary_image, perspective_matrix, (image.shape[1], image.shape[0]))
    # cv2.imshow('Warped Image', binary_image)
    def sliding_window(binary_image, n_windows=9, margin=200, min_pixels=100):
      # Set height of windows
      window_height = binary_image.shape[0] // n_windows

      # Identify the x and y positions of all nonzero pixels in the image
      nonzero = binary_image.nonzero()
      nonzeroy = np.array(nonzero[0])
      nonzerox = np.array(nonzero[1])

      # Current positions to be updated later for each window
      leftx_current = int(binary_image.shape[1] * 0.15)
      rightx_current = int(binary_image.shape[1] * 0.85)

      # Create empty lists to receive left and right lane pixel indices
      left_lane_inds = []
      right_lane_inds = []
      for window in range(n_windows):
          # Identify window boundaries
          win_y_low = binary_image.shape[0] - (window + 1) * window_height
          win_y_high = binary_image.shape[0] - window * window_height

          # Identify window boundaries for x-axis
          win_xleft_low = max(0, leftx_current - margin)
          win_xleft_high = min(binary_image.shape[1], leftx_current + margin)
          win_xright_low = max(0, rightx_current - margin)
          win_xright_high = min(binary_image.shape[1], rightx_current + margin)

          # Identify the nonzero pixels within the window
          good_left_inds = (
              (nonzeroy >= win_y_low) &
              (nonzeroy < win_y_high) &
              (nonzerox >= win_xleft_low) &
              (nonzerox < win_xleft_high)
          ).nonzero()[0]
          good_right_inds = (
              (nonzeroy >= win_y_low) &
              (nonzeroy < win_y_high) &
              (nonzerox >= win_xright_low) &
              (nonzerox < win_xright_high)
          ).nonzero()[0]

          # Append these indices to the lists
          left_lane_inds.append(good_left_inds)
          right_lane_inds.append(good_right_inds)

          # If enough pixels are found, recenter the next window on their mean position
          if len(good_left_inds) > min_pixels:
              leftx_current = int(np.mean(nonzerox[good_left_inds]))
          if len(good_right_inds) > min_pixels:
              rightx_current = int(np.mean(nonzerox[good_right_inds]))

      # Concatenate the arrays of indices
      left_lane_inds = np.concatenate(left_lane_inds)
      right_lane_inds = np.concatenate(right_lane_inds)
      return left_lane_inds, right_lane_inds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # binary_image = 5 * binary_image - (np.roll(binary_image, 1, axis=0) + np.roll(binary_image, -1, axis=0) +
    #                           np.roll(binary_image, 1, axis=1) + np.roll(binary_image, -1, axis=1))
    #binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    binary_image = cv2.erode(binary_image, kernel, iterations=3)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)


    threshold_value = 254  # Adjust this threshold as needed
    _, binary_result = cv2.threshold(binary_image, threshold_value, 255, cv2.THRESH_BINARY)


    left_lane_inds, right_lane_inds = sliding_window(binary_image)
    out_img = np.dstack((binary_image, binary_image, binary_image))
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Moving Average - Apply a moving average window
    window_size = 25  # Window size for moving average
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    np.save('rightx.npy', rightx)
    np.save('righty.npy',righty)
    def model_f(x, a, b, c):
      return a*(x-b)**2 + c

    popt_l, pcov_l = curve_fit(model_f, leftx, lefty, p0=[3,2,-16])
    a_opt_l, b_opt_l, c_opt_l = popt_l
    
    x_model_l = np.linspace(min(leftx), max(leftx), 1000)

    y_model_l = model_f(x_model_l, a_opt_l, b_opt_l, c_opt_l)
    x_model_l = np.int32(x_model_l)
    y_model_l = np.int32(y_model_l)
    coordinates_l = [(x, y) for x, y in zip(x_model_l, y_model_l)]
    clamped_points_l = []
    for x, y in coordinates_l:
      if (x> 0 and x<image_width -1) and (y>0 and y<image_height):
        clamped_points_l.append((x, y))
    x_coordinates_l = [x for x, _ in clamped_points_l]
    y_coordinates_l = [y for _, y in clamped_points_l]
    try:
      popt_r, pcov_r = curve_fit(model_f, rightx, righty, p0=[3,2,-16])
      a_opt_r, b_opt_r, c_opt_r= popt_r
      x_model_r = np.linspace(min(rightx), max(rightx), 1000)

      y_model_r = model_f(x_model_r, a_opt_r, b_opt_r, c_opt_r)
      x_model_r = np.int32(x_model_r)
      y_model_r = np.int32(y_model_r)
      print(y_model_r)
      coordinates_r = [(x, y) for x, y in zip(x_model_r, y_model_r)]
      clamped_points_r = []
      for x, y in coordinates_r:
        if (x> 0 and x<image_width -1) and (y>0 and y<image_height):
          clamped_points_r.append((x, y))
      x_coordinates_r = [x for x, _ in clamped_points_r]
      y_coordinates_r = [y for _, y in clamped_points_r]
    except:
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # Mark left lane pixels in red
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # Mark right lane pixels in blue
      out_img[y_coordinates_l, x_coordinates_l] = [0, 255, 0]  # Mark left lane pixels in red



    # Get the inlier mask and fitted polynomial coefficients
    #inlier_mask = ransac.inlier_mask_
    #poly_coefficients = ransac.estimator_.coef_
    # Extract the inlier points for visualization or further analysis
    #inlier_points = leftx[inlier_mask], lefty[inlier_mask]

    # # Display the curve image
    # cv2.imshow('Fitted Curve', curve_image)
    # # Retrieve the best line parameters
    # slope = ransac.estimator_.coef_[0][0]
    # intercept = ransac.estimator_.intercept_[0]

    # # Calculate the endpoints of the line
    # x1 = 0
    # y1 = int(intercept)
    # x2 = binary_image.shape[1]
    # y2 = int(slope * x2 + intercept)

    # # Plot the fitted line on the binary image
    # line_image = np.zeros_like(binary_image)
    # cv2.line(line_image, (x1, y1), (x2, y2), color=255, thickness=2)
    # cv2.imshow('test',line_image)
    
    # left_fit = np.polyfit(lefty, leftx, 2)
    # ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 +left_fit[1]*ploty + left_fit[2]
	  # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	  #right_fit = np.polyfit(righty, rightx, 2)
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # Mark left lane pixels in red
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # Mark right lane pixels in blue
    # out_img[y_coordinates_l, x_coordinates_l] = [0, 255, 0]  # Mark left lane pixels in red
    # out_img[y_coordinates_r,x_coordinates_r] = [0,255,0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # Mark right lane pixels in blue
    # Display the original image with selected points and the warped image
    #cv2.imshow('Image with Points', image)
    cv2.imshow('Warped Image', out_img)
  #  cv2.imshow('unwarped_image',unwarped_image)
    cv2.waitKey(0)

# Close all open windows
cv2.destroyAllWindows()
