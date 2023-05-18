import numpy as np
import cv2
import pandas as pd
import csv


df =  pd.read_csv("mm_to_ml - Sheet1 (1).csv")
urine_height_mm = df["mm"].values
ml_height = df["ml"].values

mf = 0

# Create an empty dictionary to store mm-to-ml mappings
mm_to_ml = {}
distances = []
window_size = 40
# Define the yellow color range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Define the blue color range
lower_blue = np.array([20, 40, 50])
upper_blue = np.array([140, 255, 255])

yellow_height = 0
blue_height = 0
# max_blue_height = 0
total_height = 115.69 
# csv_path = "mm_to_ml - Sheet1 (1).csv"


def closest_index_array(arr, num):
    index = (np.abs(arr - num)).argmin()
    return index


def closest_index_list(lst, num):
    closest_val = min(lst, key=lambda x: abs(x - num))
    return lst.index(closest_val)


def get_urine_volume(img, cls=0.0):
    print("first_Second is called")
    if cls == 0.0:
        print("Small Box")
        # print(img)
        small_box(img)
        # print("Large Box")
        # # print(img)
        # large_Box(img) 


def get_urine_volume2(img, cls=0.0):
    print("for larg Second is called")
    if cls == 0.0:
        print("Large Box")
        # print(img)
        large_Box(img)        


def small_box(frame):
    global mf
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological operations to remove small shapes
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Detect contours in the image
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw only vertical blue lines
    max_h = 0

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)


        if w <<h:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # print("width", w)
            print("height", h)
            if h > max_h:
                max_h = h
    print("Maximum h:", max_h)

    # Display images
    cv2.imshow('webcam', frame)
    cv2.imshow('opening',opening)
    cv2.imshow ('closing',closing)
    


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
        print("Length of yellow height: ",distance)
        distances.append(distance)
        
        # Calculate the average distance
        average_distance = round(sum(distances) / len(distances))
        cv2.imshow('frame_s', frame)
        mm_height = (116.5/max_h)*distance
        mf = (116.5/max_h)
        print("mf:" ,mf)
        print("volume_urine_mm: ",mm_height)
        # cv2.waitKey(0)

        urine_volume = ml_height[closest_index_array(mm_height, urine_height_mm)]
        print(f"Urine Height in (mm): {mm_height} -> Urine Volume in Small Box: {urine_volume} (ml)")

    
def large_Box(frame):
        # Define the yellow color range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([30, 255, 255]) 

        # print("mf2",mf)
        yellow_height = 0
        # mf = 0

        # Measure the height of yellow color here

        # Update the yellow_height variable with the measured value
        yellow_height = yellow_height

        
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
            mm = mf*h
            print("Yellow_height_large_Box_mm:",mm)
            print("yellow height in large box:",yellow_height)

        # Display the result
        cv2.imshow('yellow', yellow_mask)

        # Display the measurement
        # cv2.putText(frame, "Yellow height: {}".format(yellow_height), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Measurement', frame)
         # Exit if 'q' is pressed
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     # break
        #         cv2.destroyAllWindows()



def get_urine_volume(img, cls=0.0):
    print("Second is called")
    if cls == 0.0:
        print("Small Box")
        # print(img)
        small_box(img)
    else:
        print("large Box")
        # print(img)
        large_Box(img)