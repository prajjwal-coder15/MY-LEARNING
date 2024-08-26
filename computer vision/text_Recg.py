import cv2
import numpy as np


def text_detection(s, e_p, min_confidence=0.5):
    # Load the input image
    image = cv2.imread(s)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Set the new width and height
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # Resize the image
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define the two output layers
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet(e_p)

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass
    (scores, geometry) = net.forward(layerNames)

    # Grab the number of rows and columns from the scores volume
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the columns
        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.3)

    # Loop over the bounding boxes
    for i in boxes.flatten():
        (startX, startY, endX, endY) = rects[i]

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)


# Usage
image_path = "image4.jpg"
east_path = 'models/frozen_east_text_detection.pb'
text_detection(image_path, east_path)
