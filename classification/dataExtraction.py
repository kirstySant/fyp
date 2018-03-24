import matplotlib.pyplot as plt
#import matplotlib.image as img
import skimage.feature as skimg
import skimage.measure as skm 		#entropy function
import numpy as np
import cv2
import glob

###ASK ABOUT TEST CASES

from scipy.stats import kurtosis, skew

data = np.empty(1, 20)

image = cv2.imread('/home/kirsty/Desktop/TEST/ich_1.jpg', cv2.IMREAD_GRAYSCALE)
image_flat = np.copy(image).flatten()
plt.figure()
plt.imshow(image, cmap='gray', interpolation = 'bicubic')
plt.show()



