def large_Box(frame):
    # Define the yellow color range
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Define the blue color range
    lower_blue = np.array([20, 40, 50])
    upper_blue = np.array([140, 255, 255])

    yellow_height = 0
    blue_lines = 0

    # Measure the height of yellow color here

    # Update the yellow_height variable with the measured value
    yellow_height = yellow_height
    frame = cv2.imread('uri.jpg')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Create a mask for the blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the masks to the frame
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)

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
        print("yellow height in large box:",yellow_height)

    # Find contours in the blue mask
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the contour is horizontal and long enough to be a line
            if h < w and w > 50:
                # Draw the bounding rectangle on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Increment the blue_lines counter
                blue_lines += 1
        print("Number of horizontal blue lines in large box:", blue_lines)

    # Display the result
        cv2.imshow('yellow', yellow_mask)
        cv2.imshow('blue', blue_mask)

    # Display the measurement
    # cv2.putText(frame, "Yellow height: {}".format(yellow_height), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Measurement', frame)

    Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        # break
            cv2.destroyAllWindows