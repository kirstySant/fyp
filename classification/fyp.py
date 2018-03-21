import matplotlib.pyplot as plt
import matplotlib.image as img
import skimage.feature as skimg
import skimage.measure as skm 		#entropy function
import numpy as np
import cv2
import glob

###ASK ABOUT TEST CASES

from scipy.stats import kurtosis, skew

##=========NOTES=========##

##=======================##

#-----------------------------------------------------------------------------------------------------------------------------------------
#DATA FETCHING
#get the path where all iages are being stored, similar to  johm's procedure in CPP
pathname = "/home/kirsty/Desktop/GIT_FYP/fyp/classification"
filenames = glob.glob(pathname)

print(filenames)
print(glob.glob(pathname))
#-----------------------------------------------------------------------------------------------------------------------------------------
#FEATURE EXTRACTION
#obtain numeric values for the image, thus this function would contain all extractions
#store extractions in an array that will later be appended to the matrix of input data
#or define global matrix for inputs and store directly there <- best option
#-----------------------------------------------------------------------------------------------------------------------------------------




##IMAGE FEATURES - intensity features from grayscale original image
def getVariance(x):
	return np.var(x) / x.size

def getMean(x):
	return x.mean() / x.size

def getKurtosis(x):
	return kurtosis(x)

def getSkewness (x):
 	return skew(x)

##==================================================================================
##texture features - gray level cooccurrence matrix 
#these require calculations along 4 spatial orientations (0, 45, 90 and 135 degrees) 
#and then take the average of all 4 so that values are rotation invariant

#size of GLCM == number of shades present, assumed to be 255 (all shades of gray)
def getContrast(x):
	value=skimg.greycoprops(x, 'contrast').mean()
	print("CONTRAST: "+str(value))
	print(skimg.greycoprops(x, 'contrast'))
	print("+++++++++++++++++++++++++++++++++++")
	return skimg.greycoprops(x, 'contrast')

def getHomogeneity(x):
	value = skimg.greycoprops(x, 'homogeneity').mean()
	print("HOMOGENEITY: "+str(value))
	print(skimg.greycoprops(x, 'homogeneity'))
	print("+++++++++++++++++++++++++++++++++++")
	return skimg.greycoprops(x, 'homogeneity')

def getASM(x):
	value = skimg.greycoprops(x, 'ASM').mean()
	print("ASM: "+ str(value))
	print(skimg.greycoprops(x, 'ASM'))
	print("+++++++++++++++++++++++++++++++++++")
	return skimg.greycoprops(x, 'ASM')

def getCorrelationCoeff(x):
	value = skimg.greycoprops(x, 'correlation').mean()
	print("CORRELATION COEFFICIENT: " + str(value))
	print(skimg.greycoprops(x, 'correlation'))
	print("+++++++++++++++++++++++++++++++++++")
	return skimg.greycoprops(x, 'correlation')


def getEnergy(x):
	value = skimg.greycoprops(x, 'energy').mean()
	print("ENERGY: "+str(value))
	print(skimg.greycoprops(x, 'energy'))
	print("+++++++++++++++++++++++++++++++++++")
	return skimg.greycoprops(x, 'energy')


#calculating the entropy
def getEntropy(x):
	entropy0 = skm.shannon_entropy(x[:,:,0,0])
	entropy45 = skm.shannon_entropy(x[:,:,0,1])
	entropy90 = skm.shannon_entropy(x[:,:,0,2])
	entropy135 = skm.shannon_entropy(x[:,:,0,3])
	entropy = (entropy0 + entropy45 + entropy90 + entropy135) / 4
	print("ENTROPY: "+str(entropy))
	return entropy




#pixels = img.imread('puppy.jpg')
#print(pixels)
image = cv2.imread('thresh_35.png', cv2.IMREAD_GRAYSCALE) #note that cv2.IMREAD_GRAYSCALE == 0
print("____________________________________________________")
print(image)

image_flat = np.copy(image).flatten()
print("____________________________________________________")
#image_flat.flatten()
print(image_flat)
print("____________________________________________________")

print("variance: ")
print(getVariance(image))
print("mean: ")
print(getMean(image))
print("kurtosis: ")
print(getKurtosis(image_flat))
print("skewness: ")
print(getSkewness(image_flat))

skGLCM = skimg.greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed="true")
print("=======================================================")
print("0 deg")
glcm0 = skGLCM[:,:,0,0]
print(glcm0)
print("=======================================================")
print("45 deg")
glcm45 = skGLCM[:,:,0,1]
print(glcm45)
print("=======================================================")
print("90 deg")
glcm90 = skGLCM[:,:,0,2]
print(glcm90)
print("=======================================================")
print("135 deg")
glcm135 = skGLCM[:,:,0,3]
print(glcm135)
print("=======================================================")


getHomogeneity(skGLCM)
getASM(skGLCM)
getCorrelationCoeff(skGLCM)
getEnergy(skGLCM)
getContrast(skGLCM) #contrast giving me value that is not normalised. i dont know why.
getEntropy(skGLCM)

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
#NEURAL NETWORK PART - TRINING ONLY

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x) 

in_neurons = 20
l1_neurons = 12
l2_neurons = 6
out_neurons = 5
samplesize = 100	#arbitrarily chosen
learningRate = 0.1 	#arbitrarily chosen


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
