
import cv2
import numpy as np # type: ignore
      # Define a function to convert pixel values to milliliters
def convert_pixels_to_ml(average_distace):
    # Find the index of the closest pixel value in the pixels array
        index = np.abs(distance - average_distance).argmin()
    # # Return the corresponding milliliter value from the milliliters array
        return milliliters[index]


# pixel values,
average_dis= np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
              19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
              35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
              51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 
              67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
              83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 
              99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 
              112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
              124, 125, 126, 127, 128, 129, 130, 131, 132, 
              133, 134, 135, 136, 137, 138, 139, 140, 141, 
              142, 143, 144, 145, 146, 147, 148, 149, 150, 
              151, 152, 153, 154, 155, 156, 157, 158, 159, 
              160, 161, 162, 163, 164, 165, 166, 167, 168, 
              169, 170, 171, 172, 173, 174, 175, 176, 177, 
              178, 179, 180, 181, 182, 183, 184, 185, 186, 
              187, 188, 189, 190, 191, 192, 193, 194, 195, 
              196, 197, 198, 199, 200, 201, 202, 203, 204, 
              205, 206, 207, 208, 209, 210, 211, 212, 213, 
              214, 215, 216, 217, 218, 219, 220, 221, 222, 
              223, 224, 225, 226, 227, 228, 229, 230, 231, 
              232, 233, 234, 235, 236, 237, 238, 239, 240, 
              241, 242, 243, 244, 245, 246, 247, 248, 249, 
              250, 251, 252, 253, 254, 255, 256, 257, 258, 
              259, 260, 261])


# milliliter values
milliliters = np.array([1, 1.5, 2, 2.25, 2.5, 3, 3.25, 3.5, 4, 4.333, 4.666, 5, 5.41, 5.82, 
              6.23, 6.64, 7.05, 7.46, 7.87, 8.28, 8.69, 9.1, 9.51, 10, 10.45 ,
              11.35, 11.8, 12.25, 12.7, 13.15, 13.6, 14.5, 14.95, 15.4, 15.85 ,
              16.169, 16.4882, 16.8074, 17.1266, 17.4458, 17.765, 18.0842, 18.4034, 18.7226, 
              19.0418, 19.361, 20, 20.384615, 20.76923, 21.153845, 21.53846, 21.923075, 22.30769, 
              22.692305, 23.07692, 23.461535, 23.84615, 24.230765, 25, 25.625, 26.25, 26.875, 27.5, 
              28.125, 28.75, 29.375, 30, 30.625, 31.25, 31.875, 32.5, 33.125, 33.75, 34.375, 35, 
              35.625, 36.25, 36.875, 37.5, 38.125, 38.75, 39.375, 40, 40.625, 41.25, 41.875, 42.5, 
              43.125, 43.75, 44.375, 45, 45.625, 46.25, 46.875, 47.5, 48.125, 48.75, 49.375, 50, 
              50.71428, 51.42856, 52.14284, 52.85712, 53.5714, 54.28568, 54.99996, 55.71424, 56.42852, 
              57.1428, 57.85708, 58.57136, 59.28564, 59.99992, 60.7142, 61.42848, 62.14276, 62.85704, 
              63.57132, 64.2856, 64.99988, 65.71416, 66.42844, 67.14272, 67.857, 68.57128, 69.28556, 
              69.99984, 70.71412, 71.4284, 72.14268, 72.85696, 73.57124, 74.28552, 74.9998, 75.71408, 
              76.71408, 77.71408, 78.71408, 79.71408, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 
              92, 93, 94, 95, 95.8333, 96.6666, 97.4999, 98.3332, 99.1665, 100, 101.25, 102.5, 103.75, 
              105, 106.25, 107.5, 108.75, 110, 111.25, 112.5, 113.75, 115, 116.25, 117.5, 118.75, 120, 
              121.25, 122.5, 123.75, 125, 126.25, 127.5, 128.75, 130, 131.25, 132.5, 133.75, 135, 136, 
              137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151.25, 152.5, 153.75, 
              155, 156.66666, 158.33332, 159.99998, 161.66664, 163.3333, 164.99996, 166.24996, 167.49996, 
              168.74996, 169.99996, 171.66596, 173.33196, 174.99796, 176.24796, 177.49796, 178.74796, 179.99796, 
              181.66396, 183.32996, 184.99596, 186.34596, 187.69596, 189.04596, 190.24596, 191.54596, 192.84596, 
              194.14596, 195.44596, 196.74596, 198.04596, 199.34596, 202.04596, 204.74596, 207.44596, 210.14596, 
              212.64596, 215.14596, 217.64596, 220.14596, 222.64596, 225.14596, 227.64596, 230.14596, 232.64596, 
              235.14596, 237.64596, 240.14596, 242.64596, 245.14596, 247.64596, 250.14596, 252.64596])
   

# Set up the video capture device
# cap = cv2.VideoCapture(0)
distances = []
window_size = 40
# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
frame = cv2.imread("ur.jpg")
    
    # Convert the frame to the HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the lower and upper range of the yellow color in HSV
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# Define the blue color range
lower_blue = np.array([35, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create a binary mask by thresholding the frame based on the yellow color range
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
kernel = np.ones((5,5),np.uint8)# 5x5 kernel with full of ones. 
e = cv2.erode(mask,kernel) #optional parameters   iterations = 10
kernel = np.ones((5,5),np.uint8)# 5x5 kernel with full of ones.  
d = cv2.dilate(mask,kernel) #iterations = 2 (optional parameters) iterations = 10
# Apply the masks to the frame
blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
cv2.imshow("dilate", d)
edges = cv2.Canny(mask,50,200)
cv2.imshow("edges",edges)

# Apply edge detection on the blue mask
blue_edges = cv2.Canny(blue_mask, 50, 150)
# Apply Hough transform to detect lines in the blue mask
lines = cv2.HoughLinesP(blue_edges, 1, np.pi/180, 50, minLineLength=15, maxLineGap=20)
# Draw the detected lines on the original frame and calculate vertical length of blue line
blue_length = 0
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if x1 == x2:
            blue_length = abs(y2 - y1)
            print(blue_length)


# # Find the contours in the binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # Initialize an empty list to store the distance values


# Find the contour with the largest area, which should be the liquid in the box
if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the coordinates of the top and bottom points of the contour
    top_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    bottom_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    # Draw lines at the top and bottom coordinates of the contour on the original frame
    cv2.line(frame, (top_point[0], top_point[1]), (top_point[0] + frame.shape[1], top_point[1]), (0, 255, 0), 2)
    cv2.line(frame, (bottom_point[0], bottom_point[1]), (bottom_point[0] + frame.shape[1], bottom_point[1]), (0, 255, 0), 2)
    y1 = top_point[1]
    y2 = bottom_point[1]
    distance = abs(y2 - y1)
    distances.append(distance)
    
    # Calculate the average distance
    average_distance = round(sum(distances) / len(distances))
    # Calculate the average distance only if the distances list is not empty
    # 
    # s.pop(0)
    # print(average_distance)
    # average_distance = calculate_average_distance(pixels)
    # ml_value = milliliters[average_distance-2]
    # print(f"A pixel value of {average_distance} corresponds to {ml_value} ml.")  


# Display the frame
cv2.imshow('frame', frame)
# cv2.imshow('edges',edge)

# Exit if 'q' is pressed
if cv2.waitKey(0) & 0xFF == ord('q'):
    # break

# Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()