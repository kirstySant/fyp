import matplotlib.pyplot as plt
#import matplotlib.image as img
import skimage.feature as skimg
import skimage.measure as skm 		#entropy function
import numpy as np
import cv2
import glob

###ASK ABOUT TEST CASES

from scipy.stats import kurtosis, skew

##=========NOTES=========##

##=======================##
#global definitions
in_neurons = 20
l1_neurons = 12
l2_neurons = 6
out_neurons = 5
samplesize = 100	#arbitrarily chosen
learningRate = 0.1 	#arbitrarily chosen


inputMatrixSize = (samplesize, in_neurons)
inputMatrix = np.empty(inputMatrixSize)

def getGrayscaleFeatures(im, sample):
    
    #intensity features
    im_flat = np.copy(im).flatten()     #flatten to be able to compute skewness and kurtosis
    inputMatrix[sample, 0] = im.mean() / im.size
    inputMatrix[sample, 1] = np.var(im) / im.size
    inputMatrix[sample, 2] = skew(im_flat)
    inputMatrix[sample, 3] = kurtosis(im_flat)
    
    #texture features:
    ##texture features - gray level cooccurrence matrix 
    #these require calculations along 4 spatial orientations (0, 45, 90 and 135 degrees) 
    #and then take the average of all 4 so that values are rotation invariant
    #size of GLCM == number of shades present, assumed to be 255 (all shades of gray)
    GLCM = skimg.greycomatrix(im,                                               #image 
                              [1],                                              #pixel offsets, 1 implies 1 pixel shift each time
                              [0, np.pi/4, np.pi/2, 3*np.pi/4],                 #angles in radians - 4 given to make it rotational invariant
                              normed="true")                                    #normalised values, sum of matrix results to 1
    inputMatrix[sample, 4] = skimg.greycoprops(GLCM, 'contrast').mean()
    inputMatrix[sample, 5] = skimg.greycoprops(GLCM, 'homogeneity').mean()
    inputMatrix[sample, 6] = skimg.greycoprops(GLCM, 'ASM').mean()
    inputMatrix[sample, 7] = skimg.greycoprops(GLCM, 'correlation').mean()
    inputMatrix[sample, 8] = skimg.greycoprops(GLCM, 'energy').mean()
    entropy = ((skm.shannon_entropy(GLCM[:,:,0,0])) + 
               (skm.shannon_entropy(GLCM[:,:,0,1])) + 
               (skm.shannon_entropy(GLCM[:,:,0,2])) + 
               (skm.shannon_entropy(GLCM[:,:,0,3]))) / 4
    inputMatrix[sample, 9] = entropy


#-----------------------------------------------------------------------------------------------------------------------------------------
#DATA FETCHING
#get the path where all iages are being stored, similar to  johm's procedure in CPP
pathname_grayscale = "/home/kirsty/Desktop/GIT_FYP/fyp/classification/Training/grayscale/*.png" # refine path name!!!
filenames_grayscale = sorted(glob.glob(pathname_grayscale))

#print(filenames)
for i in range(0, len(filenames_grayscale)):
    currentImagePath = filenames_grayscale[i]
    #print(currentImagePath)
    #printing all images to check they are ok
    image = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
    plt.figure(i+1)
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    getGrayscaleFeatures(image, i)
    #print("image "+str(i))
    #print(inputMatrix[:i])
#plt.show()

pathname_contour = "/home/kirsty/Desktop/GIT_FYP/fyp/classification/Training/contours*.png" #change path name!!!
filenames_contour = sorted(glob.glob(pathname_contour))

for j in range(0, len(filenames_contour)):
    currentImagePath = filenames_contour[i]
    image_contour = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
    plt.figure(i+len(filenames_grayscale))
    plt.imshow(image_contour, cmap='gray', interpolation='bicubic')
    
    #print("image "+str(j))
    #print(inputMatrix[:j])
#plt.show()

print(inputMatrix)
#saving to file for testing purposes. 
np.savetxt('inputMatrixTest.txt', inputMatrix, fmt='%.2f')


#-----------------------------------------------------------------------------------------------------------------------------------------
#FEATURE EXTRACTION
#obtain numeric values for the image, thus this function would contain all extractions
#store extractions in an array that will later be appended to the matrix of input data
#or define global matrix for inputs and store directly there <- best option
#-----------------------------------------------------------------------------------------------------------------------------------------


def getShapeFeatures(im, sample):
    inputMatrix(sample, 10 = )



#pixels = img.imread('puppy.jpg')
#print(pixels)
image = cv2.imread('thresh_35.png', cv2.IMREAD_GRAYSCALE) #note that cv2.IMREAD_GRAYSCALE == 0
print("____________________________________________________")
print(image)



ret, thresh = cv2.threshold(image, 127,255,0)
image2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.figure(1)
plt.imshow(image, cmap='gray', interpolation='bicubic')

plt.figure(2)
for count in contours:
	f=count+2
	plt.figure()
	cv2.drawContours(image2, [count], 0, (0,255,0), 3)

plt.imshow(image2, cmap='gray', interpolation='bicubic')
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------
#NEURAL NETWORK PART - TRAINING ONLY

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x) 

np.random.seed(1)

X = np.random.randint(2, size=(samplesize, in_neurons))
Y = np.random.randint(2, size=(samplesize, out_neurons))

w0 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
w1 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
w2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1

print(X)
print ("\n\n")
print(w0)
print ("\n\n")
print(w1)
print ("\n\n")
print(w2)
print ("\n\n")
print(Y)
for i in range(0, 1000):	#chosen arbitrarily
	l0 = X
	l1 = sigmoid(np.dot(l0, w0))
	l2 = sigmoid(np.dot(l1, w1))
	l3 = sigmoid(np.dot(l2, w2))

	l3_error = Y - l3 								#error in output
	l3_delta = l3_error * sigmoid_derivative(l3)	#delta for output layer, will be used to alter the weights

	l2_error = np.dot(l3_delta, w2.T)
	l2_delta = l2_error * sigmoid_derivative(l2)

	l1_error = np.dot(l2_delta, w1.T)
	l1_delta = l1_error * sigmoid_derivative(l1)

	w2 += np.dot(l2.T, l3_delta) * learningRate
	w1 += np.dot(l1.T, l2_delta) * learningRate
	w0 += np.dot(l0.T, l1_delta) * learningRate

print("AFTER FOR LOOP:\n\n")
print(l0)
print ("\n\n")
print(l1)
print ("\n\n")
print(l2)
print ("\n\n")
print(l3)
print ("\n\n")
print("percentage error (final) "+str(np.mean(l3_error)))
#-----------------------------------------------------------------------------------------------------------------------------------------
