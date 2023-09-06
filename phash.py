import numpy as np
import cv2

# def map_value(x, x_min, x_max, y_min, y_max):
#     return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

# def map_img(image : np.ndarray):
#     rt_img = np.zeros(image.shape, dtype=np.uint8)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rt_img[i, j] = int(map_value(image[i, j], image.min(), image.max(), 0, 255))
#     return rt_img

def phash(image : np.ndarray, hash_size=8):
    # Convert the array value into [0, 255]
    image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)
    # Resize the image to the desired hash size + 1x1
    resized = cv2.resize(image, (32, 32))

    imdct = cv2.dct(np.float32(resized))
    imdct = imdct[:hash_size, :hash_size]
    mid = np.median(imdct)
    diff = imdct > mid

    # Convert the diff array to a bitstring and then to an integer
    bitstring = np.packbits(diff.flatten())
    hash_int = 0
    for bit in bitstring:
        hash_int = (hash_int << 1) | bit

    return hash_int