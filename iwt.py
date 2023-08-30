import numpy as np
import pywt
import cv2

def image_to_iwt(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform the 2D discrete wavelet transform
    coeffs = pywt.dwt2(image, 'haar')

    # The result is a tuple of (approximation, (horizontal, vertical, diagonal))
    approximation, (horizontal, vertical, diagonal) = coeffs

    # Convert the approximation and details coefficients to integers
    approximation = np.round(approximation).astype(int)
    horizontal = np.round(horizontal).astype(int)
    vertical = np.round(vertical).astype(int)
    diagonal = np.round(diagonal).astype(int)

    # Return the coefficients
    return approximation, (horizontal, vertical, diagonal)

def iwt_to_image(approximation, details):
    # Convert the approximation and details coefficients to floats
    approximation = approximation.astype(float)
    horizontal, vertical, diagonal = details
    horizontal = horizontal.astype(float)
    vertical = vertical.astype(float)
    diagonal = diagonal.astype(float)

    # Perform the inverse 2D discrete wavelet transform
    image = pywt.idwt2((approximation, (horizontal, vertical, diagonal)), 'haar')

    # Convert the image to integers
    image = np.round(image).astype(int)

    # Return the image
    return image