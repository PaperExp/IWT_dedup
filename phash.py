import numpy as np
import cv2

def phash(image, hash_size=8):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired hash size + 1x1
    resized = cv2.resize(gray, (hash_size + 1, hash_size))

    # Compute the horizontal gradient between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert the binary array to a hash string
    return np.packbits(diff.flatten()).tolist()