# import cv2
# import matplotlib.pyplot as plt


# # Load the image
# img = cv2.imread('roi.jpg', 0)

# plt.imshow(img)
# plt.show()
# # Define the ROI for the small box
# sx, sy, sx1, sy1 = 459, 281, 350, 123

# # Define the ROI for the large bag
# x2, y2, w2, h2 = 458, 97, 64, 328

# # Crop the small box
# crop_small_box = img[sy1:sy, sx1:sx]

# # Crop the large bag
# # crop_large_bag = img[y2:y2+h2, x2:x2+w2]

# # Display the cropped images
# cv2.imshow('Small Box', crop_small_box)
# # cv2.imshow('Large Bag', crop_large_bag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import pandas as pd


# def closest_index_list(lst, num):
#     closest_val = min(lst, key=lambda x: abs(x - num))
#     return lst.index(closest_val)


# def closest_index_array(arr, num):
#     index = (np.abs(arr - num)).argmin()
#     return index

# df = pd.read_csv("mm_to_ml.csv")
# mm_height = df["mm"].values
# ml_height = df["ml"].values

# # Define the yellow color range
# lower_yellow = np.array([15, 100, 100])
# upper_yellow = np.array([30, 255, 255])

# # Define the blue color range
# lower_blue = np.array([80, 50, 50])
# upper_blue = np.array([140, 255, 255])

# yellow_height = 0
# blue_height = 0
# max_blue_height = 0
# total_height = 115.69 

# # Start the camera
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
#     # Resize the frame to 480x360
#     # frame = cv2.resize(frame, (480, 360))
#     # out.write(resized_frame)

#     # Convert the frame to the HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Create a mask for the yellow color
#     yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     # Apply the mask to the frame
#     yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

#     # Find contours in the yellow mask
#     contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Find the largest contour in the yellow mask
#         largest_contour = max(contours, key=cv2.contourArea)
#         # Get the bounding rectangle of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         # Draw the bounding rectangle on the frame
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         # Calculate the vertical height of the yellow color
#         yellow_height = h
#         # print(yellow_height)

#     # Create a mask for the blue color
#     blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

#     # Apply the mask to the frame
#     blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)

#     # Find contours in the blue mask
#     contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Loop through the contours and find the one with the largest height-to-width ratio
#     max_ratio = 0
#     max_contour = None
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 0:
#             ratio = h / w
#             if ratio > max_ratio and ratio > 4:  # set a threshold for the ratio
#                 max_ratio = ratio
#                 max_contour = contour

#     if max_contour is not None:
#         # Get the bounding rectangle of the largest contour
#         x, y, w, h = cv2.boundingRect(max_contour)
#         # Draw the bounding rectangle on the frame
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         # Calculate the vertical height of the blue color
#         blue_height = h
#         # Update maximum height found so far
#         if blue_height > max_blue_height:
#             max_blue_height = blue_height
#         # print(blue_height)
#         total_height= yellow_height + max_blue_height 
#         one_pixel = 115.69/(total_height)
#         urine_height_mm = round(yellow_height*one_pixel, 3)
#         # print(yellow_height + max_blue_height)
#         urine_volume = ml_height[closest_index_array(mm_height, urine_height_mm)]
#         print(f"Urine Height in (mm): {urine_height_mm} -> Urine Volume: {urine_volume} (ml)")

#     # Display the result
#     cv2.imshow('yellow', yellow_mask)
#     cv2.imshow('blue', blue_mask)
#     # cv2.imshow('frame', frame)

#     # Display the measurements
#     cv2.putText(frame, "Yellow height: {}".format(yellow_height), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     # cv2.putText(frame, "blue_height:{}" .format(blue_height), (40, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     cv2.putText(frame, "Max blue height: {}".format(max_blue_height), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     cv2.imshow('Measurement', frame)

#     # Check for key press and break the loop if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the camera and destroy all windows
# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np
import pandas as pd


def 

(lst, num):
    closest_val = min(lst, key=lambda x: abs(x - num))
    return lst.index(closest_val)


def closest_index_array(arr, num):
    index = (np.abs(arr - num)).argmin()
    return index

df = pd.read_csv("mm_to_ml.csv")
mm_height = df["mm"].values
ml_height = df["ml"].values

# Define the yellow color range
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Define the blue color range
lower_blue = np.array([80, 50, 50])
upper_blue = np.array([140, 255, 255])

yellow_height = 0
blue_height = 0
max_blue_height = 0
total_height = 115.69 

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Resize the frame to 480x360
    # frame = cv2.resize(frame, (480, 360))
    # out.write(resized_frame)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the frame
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # Find contours in the yellow mask
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour in the yellow mask
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Draw the bounding rectangle on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Calculate the vertical height of the yellow color
        yellow_height = h
        # print(yellow_height)

    # Create a mask for the blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the frame
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Find contours in the blue mask
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find the one with the largest height-to-width ratio
    max_ratio = 0
    max_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0:
            ratio = h / w
            if ratio > max_ratio and ratio > 4:  # set a threshold for the ratio
                max_ratio = ratio
                max_contour = contour

    if max_contour is not None:
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)
        # Draw the bounding rectangle on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Calculate the vertical height of the blue color
        blue_height = h
        # Update maximum height found so far
        if blue_height > max_blue_height:
            max_blue_height = blue_height
        # print(blue_height)
        total_height= yellow_height + max_blue_height 
        one_pixel = 115.69/(total_height)
        urine_height_mm = round(yellow_height*one_pixel, 3)
        # print(yellow_height + max_blue_height)
        urine_volume = ml_height[closest_index_array(mm_height, urine_height_mm)]
        print(f"Urine Height in (mm): {urine_height_mm} -> Urine Volume: {urine_volume} (ml)")

    # Display the result
    cv2.imshow('yellow', yellow_mask)
    cv2.imshow('blue', blue_mask)
    # cv2.imshow('frame', frame)

    # Display the measurements
    cv2.putText(frame, "Yellow height: {}".format(yellow_height), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.putText(frame, "blue_height:{}" .format(blue_height), (40, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Max blue height: {}".format(max_blue_height), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Measurement', frame)

    # Check for key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
