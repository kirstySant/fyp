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
in_neurons = 18
l1_neurons = 12
l2_neurons = 6
out_neurons = 5
samplesize = 10	#arbitrarily chosen
learningRate = 0.1 	#arbitrarily chosen


inputMatrixSize = (samplesize, in_neurons)
inputMatrix = np.empty(inputMatrixSize)
#attempting to apply sigmoid function to input matrix to get normalised values (between -1 and 1)
normalisedInputMatrix = np.zeros(inputMatrixSize)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#FEATURE EXTRACTION
#obtain numeric values for the image, thus this function would contain all extractions
#store extractions in an array that will later be appended to the matrix of input data
#or define global matrix for inputs and store directly there <- best option
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def getGrayscaleFeatures(im, sample):
    
    #intensity features
    im_flat = np.copy(im).flatten()     #flatten to be able to compute skewness and kurtosis
    inputMatrix[sample, 0] = im.mean()
    inputMatrix[sample, 1] = np.var(im)
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

def getShapeFeatures(im, sample):
	###THRESHOLDING MIGHT BE REDUNDANT IF THE IMAGE BEING SENT IS ALREADY BINARY
	#threshold value 50 ma nafx minn fejn gie, trying with it
	#retval, threshold_output = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
	#print(threshold_output)
    #ret, thresh = cv2.threshold(im,127,255,0)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("no of contours: "+str(len(contours)))
    #areas = np.empty([1, len(contours)])
    originalImage = np.copy(im)
    image, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.zeros([len(contours)])
    maxArea = 0
    maxAreaLoc = -1
    maxPerim = 0

    print("in getGrayFeatures")
    for i in range(1, len(areas)):
    	areas[i] = cv2.contourArea(contours[i])
    	#print(str(i)+"-----"+str(areas[i]))
    	if(areas[i] > maxArea):
    		maxArea = areas[i]
    		maxAreaLoc = i


    maxPerim = cv2.arcLength(contours[maxAreaLoc], True)
    print(str(maxArea)+" || AT || "+str(maxAreaLoc)+" || PERIM = "+str(maxPerim))
    #print(contours)
    #printing largest contour to see which one is being given
    drawing = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(drawing, contours, maxAreaLoc, (0,255,0), 5)
    cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("contours", drawing)
    cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("original", originalImage)
    cv2.waitKey()

    #maximum closed contour is the bleed, for the cases tested so far
    #now we need to extract the required reatures, and store them in the input array

    #element 10 = area - already calculated above!
    inputMatrix[sample, 10] = maxArea
    #element 11 = perimiter - already calculated above!
    inputMatrix[sample, 11] = maxPerim
    
    boundingRect = cv2.minAreaRect(contours[maxAreaLoc])
    (_, _), (w, h), theta = cv2.minAreaRect(contours[maxAreaLoc])
    print(boundingRect)
    print(str(w)+" | "+str(h)+" || "+str(theta))

    #element 12 = major axis length
    #element 13 = minor axis length
    if(w < h):
    	inputMatrix[sample, 12] = h
    	inputMatrix[sample, 13] = w
    else:
    	inputMatrix[sample, 12] = w
    	inputMatrix[sample, 13] = h
    
    #element 14 = orientation
    inputMatrix[sample, 14] = theta

    #element 15 = extent
    boundingRectArea = w*h
    extent = maxArea / boundingRectArea
    inputMatrix[sample, 15] = extent
    
    #element 16 = solidity
    convexHullArea = cv2.contourArea(cv2.convexHull(contours[maxAreaLoc]))
    solidity = maxArea / convexHullArea
    inputMatrix[sample, 16] = solidity
    
    #element 17 = convex area
    inputMatrix[sample, 17] = convexHullArea
    
    #element 18 = equivdiameter
    ##let's forget about this for now , see what happens


#-----------------------------------------------------------------------------------------------------------------------------------------
#DATA FETCHING
#get the path where all iages are being stored, similar to  johm's procedure in CPP
pathname_grayscale = "/home/kirsty/Desktop/GIT_FYP/fyp/classification/Training/grayscale/*.png" # refine path name!!!
filenames_grayscale = sorted(glob.glob(pathname_grayscale))

#print(filenames)
for i in range(0, len(filenames_grayscale)):
    currentImagePath = filenames_grayscale[i]
    ##contourImagePath = filenames_contour[i]
    #print(currentImagePath)
    #printing all images to check they are ok
    image = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
    #contour_image = cv2.imread(contourImagePath, cv2.IMREAD_GRAYSCALE)
    plt.figure(i+1)
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    
    #plt.figure()
    #plt.imshow(contour_image, cmap='gray', interpolation='bicubic')

    getGrayscaleFeatures(image, i)
    #print("image "+str(i))
    #print(inputMatrix[:i])

pathname_contour = "/home/kirsty/Desktop/GIT_FYP/fyp/classification/Training/contours/*.png" #change path name!!!
filenames_contour = sorted(glob.glob(pathname_contour))

for j in range(0, len(filenames_contour)):
    
    currentImagePath = filenames_contour[j]
    image_contour = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
    getShapeFeatures(image_contour, j)
    plt.figure(j+len(filenames_grayscale))
    plt.imshow(image_contour, cmap='gray', interpolation='bicubic')
    
    #print("image "+str(j))
    #print(inputMatrix[:j])

#plt.show()
#print(inputMatrix)
#saving to file for testing purposes. 
np.savetxt('inputMatrixTest.txt', inputMatrix, fmt='%.4f')



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NORMALISING THE INPUT MATRIX

minColValues = np.amin(inputMatrix, axis=0)
maxColValues = np.amax(inputMatrix, axis=0)

denom = maxColValues - minColValues
print(denom)

np.savetxt('denom.txt', (maxColValues, minColValues, denom), fmt ='%.4f')

numeratorTransposed = inputMatrix.T - minColValues.T[:,None]
#print(numerator)
#np.savetxt('numerator.txt', (inputMatrix), fmt='%.4f')
np.savetxt('numerator.txt', numeratorTransposed.T, fmt='%.4f')

temp = np.divide(numeratorTransposed.T, denom)
temp2 = np.multiply(temp, 2)
normalisedInputMatrix = np.subtract(temp2, 1)
np.savetxt('normalised.txt', normalisedInputMatrix, fmt='%.4f')

#-----------------------------------------------------------------------------------------------------------------------------------------
#NEURAL NETWORK PART - TRAINING ONLY



def sigmoid_derivative(x):
	return x * (1 - x) 

np.random.seed(1)

X = np.random.randint(2, size=(samplesize, in_neurons))
Y = np.random.randint(2, size=(samplesize, out_neurons))

w0 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
w1 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
w2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1

#print(X)
#print ("\n\n")
#print(w0)
#print ("\n\n")
#print(w1)
#print ("\n\n")
#print(w2)
#print ("\n\n")
#print(Y)
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

#print("AFTER FOR LOOP:\n\n")
#print(l0)
#print ("\n\n")
#print(l1)
#print ("\n\n")
#print(l2)
#print ("\n\n")
#print(l3)
#print ("\n\n")
print("percentage error (final) "+str(np.mean(l3_error)))
#-----------------------------------------------------------------------------------------------------------------------------------------
