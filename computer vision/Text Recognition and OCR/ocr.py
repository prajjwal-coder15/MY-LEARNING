import cv2
import numpy as np
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply dilation to connect letters
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def detect_words(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Find contours
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Get the original image for drawing boxes
    image = cv2.imread(image_path)
    
    # Lists to store detected words and their positions
    detected_words = []
    word_positions = []
    
    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small contours
        if w > 20 and h > 20:
            # Extract the region of interest
            roi = processed_image[y:y+h, x:x+w]
            
            # Perform OCR on the region
            word = pytesseract.image_to_string(roi, config='--psm 10')
            
            # Clean up the detected word
            word = ''.join(c for c in word if c.isalnum())
            
            if word:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_words.append(word)
                word_positions.append((x, y, w, h))
    
    # Save the image with bounding boxes
    cv2.imwrite('detected_words.jpg', image)
    
    return detected_words, word_positions

# Path to your image file
image_path = "image4.jpg"

# Detect words
words, positions = detect_words(image_path)

# Print detected words
print("Detected words:")
for word, pos in zip(words, positions):
    print(f"Word: {word}, Position: {pos}")

print("\nProcessed image with bounding boxes saved as 'detected_words.jpg'")