# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# # Define the kernel for horizontal line detection
# kernel = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#                 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#                 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
#                 [-1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1]])


# def blue_line(img, lim):
#     # img = cv2.imread("uri.jpg")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(gray.shape)


#     # Blur the image to reduce noise
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#     # Detect edges using Canny edge detection
#     edges = cv2.Canny(blurred, 10, 25)
#     cv2.imshow("Gray", gray)

#     # Dilate the edges to make them more prominent
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = cv2.dilate(edges, kernel, iterations=2)
#     # Perform erosion on the image
#     erosion = cv2.erode(dilated, kernel, iterations=2)

#     # Display the original and eroded images
#     # cv2.imshow('Eroded Image', erosion)

#     # Perform the convolution
#     result = cv2.filter2D(erosion, -1, kernel)

#     # Threshold the result to obtain a binary image
#     thres = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#     # Display the result
#     cv2.imshow('Result', thres)

#     # creating a Sliding Window For Count Lines
#     h , w = thres.shape
#     hw = 10
#     sig = []
#     limit = lim
#     print(f"Y value from img process : {limit}")
#     for i in range(h-hw):
#         # print(thres[i:i+hw].sum())
#         if i > limit:
#             sig.append(thres[i:i+hw].mean())

#     threshold = np.average(sig)
#     peaks = signal.find_peaks(sig, height=threshold)
#     print(peaks[0])


#     print(f"Number of Line: {len(peaks[0])}")
#     # Calculate volume of liquid based on number of blue lines detected
#     volume =len(peaks[0]) * 100  # assuming each blue line corresponds to 100 ml of liquid

#     # Print the volume of liquid detected
#     print(f"Volume of liquid in large Box : {volume} ml")
#     return volume
#     return len(peaks[0])
#     plt.plot(sig)
#     plt.axhline(threshold)
#     plt.show()

    # cv2.imshow('Edges', dilated)
    # cv2.imshow('Lines', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# a = int(input("enter a number:" ))
# if(a%2 == 0):
#     print("enter number is even")
# else:
#     print("enter number is odd")

# n = int(input("enter the range"))
# for i in range(1, n+1):
#     if(i%2 == 0):
#         print(i,"evene number")
#     if(i%2 !=0):
#         print(i,"odd number")
# n = int(input("enter the range"))
# m = int(input("enter the Divisor"))
# for i in range(1, n+1):
#     if(i%m) == 0:
#         print(i,"Number divisble by m ")

# n = int(input("enter the number for table required: "))
# for n in range(1, n+1):
#     for i in range (1, 11):
#         print(n, "x", i, "=" , n*i, )
#     if(n<5):
#         print("===============") 

# a = int(input("enter a number:" ))
# if(a%2 == 0)&(a%3 == 0):
#     print(" number is divisble by 6")
# else:
#     print("number is not divisble by 6")
a = int(input("enter a number: "))
if(a%2==0):
    print("number is divisible by 2")
if(a%3==0):
    print('number is divisible by 3')