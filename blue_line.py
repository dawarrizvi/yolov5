import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# Define the kernel for horizontal line detection
kernel = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [-1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1]])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])


def blue_line(img, lim):
    # img = cv2.imread("lrg_box.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)


    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 10, 50)
    # cv2.imshow("Gray", gray)

    # Dilate the edges to make them more prominent
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # Perform erosion on the image
    erosion = cv2.erode(dilated, kernel, iterations=2)

    # Display the original and eroded images
    # cv2.imshow('Eroded Image', erosion)

    # Perform the convolution
    result = cv2.filter2D(erosion, -1, kernel)

    # Threshold the result to obtain a binary image
    thres = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Display the result 
    # cv2.imshow ('Result',thres)
    # print(f"Number of Line: {len(peaks[0])}")
    # creating a Sliding Window For Count Lines
    h , w = thres.shape
    hw = 10
    sig = []
    limit = lim
    for i in range(h-hw):
        # print(thres[i:i+hw].sum())
        if i > limit:
            sig.append(thres[i:i+hw].mean())

    threshold = np.average(sig)
    peaks = signal.find_peaks(sig, height=threshold)
    # print(peaks[0])
    # print(f"Number of Line: {len(peaks[0])}")
    plt.plot(sig)
    plt.axhline(threshold)
    plt.show()
    


    print(f" Number of Line: {len(peaks[0])}")
    # Calculate volume of liquid based on number of blue lines detected
    volume =len(peaks[0]) * 100  # assuming each blue line corresponds to 100 ml of liquid

    # Print the volume of liquid detected
    # print(f"Volume of liquid in large Box : {volume} ml")
    return volume
    return len(peaks[0])
    

    # cv2.imshow('Edges', dilated)
    # cv2.imshow('Lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


