# import the necessary packages
from matplotlib import pyplot as plt
from textblob import TextBlob
import pytesseract
import argparse
import cv2
import numpy as np

def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

# since we are using Jupyter Notebooks we can replace our argument
# parsing code with *hard coded* arguments and values
args = {
    "image": "image3.jpg",
    "lang": "hi"
}

# load the input image and convert it from BGR to RGB channel ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
results = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])

    # filter out weak confidence text localizations
    if conf > 0:
        # display the confidence and text to our terminal
        print("Confidence: {}".format(conf))
        print("Text: {}".format(text))
        print("")

        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw a bounding box around the text along
        # with the text itself
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)

# show the output image
plt_imshow("OCR Result", image)

# concatenate all the text
full_text = " ".join([results["text"][i] for i in range(0, len(results["text"])) if int(results["conf"][i]) > 0])
print("Full extracted text:")
print(full_text)