import matplotlib.pyplot as plt
import skimage.feature as skimg
import skimage.measure as skm
import numpy as np
import cv2
import glob

from scipy.stats import kurtosis, skew

#global definitions
in_neurons = 18
l1_neurons = 12
l2_neurons = 6
out_neurons = 5
samplesize = 211	#arbitrarily chosen
learningRate = 0.1
testingContour = None
contourTestImage = None
w0 = np.zeros((in_neurons, l1_neurons))
w1 = np.zeros((l1_neurons, l2_neurons))
w2 = np.zeros((l2_neurons, out_neurons))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get grayscale features
def getGrayscaleFeatures(image):
    grayscaleResults = np.zeros(10)
    image_flat = np.copy(image).flatten()#flatten to be able to compute skewness and kurtosis
    grayscaleResults[0] = image.mean()
    grayscaleResults[1] = np.var(image)
    grayscaleResults[2] = skew(image_flat)
    grayscaleResults[3] = kurtosis(image_flat)

    GLCM = skimg.greycomatrix(image,                                               #image 
                              [1],                                              #pixel offsets, 1 implies 1 pixel shift each time
                              [0, np.pi/4, np.pi/2, 3*np.pi/4],                 #angles in radians - 4 given to make it rotational invariant
                              normed="true")                                    #normalised values, sum of matrix results to 1

    grayscaleResults[4] = skimg.greycoprops(GLCM, 'contrast').mean()
    grayscaleResults[5] = skimg.greycoprops(GLCM, 'homogeneity').mean()
    grayscaleResults[6] = skimg.greycoprops(GLCM, 'ASM').mean()
    grayscaleResults[7] = skimg.greycoprops(GLCM, 'correlation').mean()
    grayscaleResults[8] = skimg.greycoprops(GLCM, 'energy').mean()
    entropy = ((skm.shannon_entropy(GLCM[:,:,0,0])) + 
               (skm.shannon_entropy(GLCM[:,:,0,1])) + 
               (skm.shannon_entropy(GLCM[:,:,0,2])) + 
               (skm.shannon_entropy(GLCM[:,:,0,3]))) / 4
    grayscaleResults[9] = entropy

    return grayscaleResults
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get contour features
def getShapeFeatures(im):
    contourResults = np.zeros(8)
    originalImage = np.copy(im)
    image, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.zeros([len(contours)])
    maxArea = 0
    maxAreaLoc = -1
    maxPerim = 0

    print("in getGrayFeatures")
    for i in range(1, len(areas)):
    	areas[i] = cv2.contourArea(contours[i])
    	if(areas[i] > maxArea):
    		maxArea = areas[i]
    		maxAreaLoc = i


    maxPerim = cv2.arcLength(contours[maxAreaLoc], True)
    print(str(maxArea)+" || AT || "+str(maxAreaLoc)+" || PERIM = "+str(maxPerim))
    #drawing = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    global testingContour
    testingContour = contours[maxAreaLoc]
    #cv2.drawContours(drawing, contours, maxAreaLoc, (0,255,0), 5)
    #cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("contours", drawing)
    #cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("original", originalImage)
    #cv2.waitKey()
    plt.close('all')
    contourResults[0] = maxArea
    contourResults[1] = maxPerim
    
    (_, _), (w, h), theta = cv2.minAreaRect(contours[maxAreaLoc])

    if(w < h):
    	contourResults[2] = h
    	contourResults[3] = w
    else:
    	contourResults[2] = w
    	contourResults[3] = h
    
    contourResults[4] = theta

    boundingRectArea = w*h
    extent = maxArea / boundingRectArea
    contourResults[5] = extent
    
    convexHullArea = cv2.contourArea(cv2.convexHull(contours[maxAreaLoc]))
    solidity = maxArea / convexHullArea
    contourResults[6] = solidity
    contourResults[7] = convexHullArea
    return contourResults
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get training set images from file
def getTrainingInputMatrix():
	pathname_grayscale = "Training/grayscale/*.png" # refine path name!!!
	filenames_grayscale = sorted(glob.glob(pathname_grayscale))
	pathname_contour = "Training/contours/*.png" #change path name!!!
	filenames_contour = sorted(glob.glob(pathname_contour))

	inputMatrix = np.empty((samplesize, in_neurons))
	for i in range(0, len(filenames_grayscale)):
		print(str(i))
		currentImagePath = filenames_grayscale[i]
		image = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
		if currentImagePath.find('EDH'):
			print(currentImagePath)
		if currentImagePath.find('SDH'):
			print(currentImagePath)
		if currentImagePath.find('ICH'):
			print(cur)
		plt.figure(i+1)
		plt.imshow(image, cmap='gray', interpolation='bicubic')

		grayscale = getGrayscaleFeatures(image)

		currentImagePath = filenames_contour[i]
		image_contour = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
		contours = getShapeFeatures(image_contour)
		plt.figure(i+len(filenames_grayscale)+1)
		plt.imshow(image_contour, cmap='gray', interpolation='bicubic')
    
		inputMatrix[i, 0:10] = grayscale
		inputMatrix[i, 10:18] = contours
	return inputMatrix
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#normalise input matrix
def normaliseMatrix(matrix):
	minColValues = np.amin(matrix, axis=0)
	maxColValues = np.amax(matrix, axis=0)

	denominator = maxColValues - minColValues
	numerator = (matrix.T - minColValues.T[:,None]).T
	temp = np.divide(numerator, denominator)
	temp2 = np.multiply(temp, 2)
	normalisedMatrix = np.subtract(temp2, 1)

	return normalisedMatrix
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get output training matrix
def getTrainingOutputMatrix():
	expectedOutputMatrix = np.zeros((samplesize, out_neurons))
	trainingExpectedResults = open("trainingExpectedOutputs.txt", "r")
	index = 0;
	for line in trainingExpectedResults:
		if(line == "EDH\n"):
			expectedOutputMatrix[index] = [1,0,0,0,0]
			#print("epidural")
		elif( line == "SDH\n"):
			expectedOutputMatrix[index] = [0,1,0,0,0]
			#print ("subdural")
		elif( line == "ICH\n"):
			expectedOutputMatrix[index] = [0,0,1,0,0]
			#print ("intracranial")
		elif( line == "IVH\n"):
			expectedOutputMatrix[index] = [0,0,0,1,0]
			#print ("intra-ventricular")
		elif( line == "NO\n"):
			expectedOutputMatrix[index] = [0,0,0,0,1]
			#print ("no hemorrhage detected")
		else:
		 	print("incorrect value in file line "+ str(index))

		index += 1
	trainingExpectedResults.close()

	return expectedOutputMatrix
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#train the neural network
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x)) 

def TrainNeuralNetwork(inputMatrix, outputMatrix):
	np.random.seed()
	X = inputMatrix
	Y = outputMatrix
	f = open("error.txt", "w+")
	global w0
	global w1
	global w2
	w0 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
	w1 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
	w2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1

	for i in range(0, 50000):
		l0 = X
		l1 = sigmoid(np.dot(l0, w0))
		l2 = sigmoid(np.dot(l1, w1))
		l3 = sigmoid(np.dot(l2, w2))

		l3_error = Y - l3 								#error in output
		f.write("Error for line "+str(i)+": "+str(np.mean(l3_error)))
		l3_delta = l3_error * sigmoid_derivative(l3)	#delta for output layer, will be used to alter the weights

		l2_error = np.dot(l3_delta, w2.T)
		l2_delta = l2_error * sigmoid_derivative(l2)

		l1_error = np.dot(l2_delta, w1.T)
		l1_delta = l1_error * sigmoid_derivative(l1)

		w2 += np.dot(l2.T, l3_delta) * learningRate
		w1 += np.dot(l1.T, l2_delta) * learningRate
		w0 += np.dot(l0.T, l1_delta) * learningRate

	print("percentage error (final) "+str(np.mean(l3_error)))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get test image from file and test input matrixs
def getTestInputMatrix():
	#possible imprivement - get image to be tested  for using glob?
	global contourTestImage
	grayscaleTestImage = cv2.imread("Testing/gs_49.png", cv2.IMREAD_GRAYSCALE)
	contourTestImage = cv2.imread("Testing/c_49.png", cv2.IMREAD_GRAYSCALE)
	grayscaleTest = getGrayscaleFeatures(grayscaleTestImage)
	contoursTest = getShapeFeatures(contourTestImage)
	testInput = np.zeros(18)
	testInput[0:10] = grayscaleTest
	testInput[10:18] = contoursTest
	return testInput
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#normalise testing matrix
def normaliseTestMatrix(matrix):
	minColValues = np.amin(matrix, axis=0)
	maxColValues = np.amax(matrix, axis=0)

	denominator = maxColValues - minColValues
	numerator = (matrix.T - minColValues.T).T
	temp = np.divide(numerator, denominator)
	temp2 = np.multiply(temp, 2)
	normalisedMatrix = np.subtract(temp2, 1)

	return normalisedMatrix
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pass test case through neural network
def TestNeuralNetwork(inputMatrix):
	global w0
	global w1
	global w2
	l0 = inputMatrix
	l1 = sigmoid(np.dot(l0, w0))
	l2 = sigmoid(np.dot(l1, w1))
	l3 = sigmoid(np.dot(l2, w2))

	print(l3)
	print("showing final result:")

	return l3
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#main function
trainingInputMatrix = getTrainingInputMatrix()
normalisedTrainingInputMatrix = normaliseMatrix(trainingInputMatrix)

trainingOutputMatrix = getTrainingOutputMatrix()

TrainNeuralNetwork(normalisedTrainingInputMatrix, trainingOutputMatrix)
print('line 272')
testInputMatrix = getTestInputMatrix()
print("line 274")
testNormalisedInputMatrix = normaliseTestMatrix(testInputMatrix)
print("line 276")
finalProbabilities = TestNeuralNetwork(testNormalisedInputMatrix)
finalPercentages = np.multiply(finalProbabilities, 100)
print(finalPercentages)
maxProb = 0
maxProbIndex = -1
index = 0
for value in finalPercentages:
	if value > maxProb:
		maxProb = value
		maxProbIndex = index
	index +=1

inferredResult = np.zeros(len(finalPercentages))
inferredResult[maxProbIndex] = 1
resultString = None
print(inferredResult)
if(inferredResult[0] == 1):
	confidence = format(finalPercentages[0], '.4f')
	resultString="EDH - " +confidence +"% confident"
elif(inferredResult[1] == 1):
	confidence = format(finalPercentages[1], '.4f')
	resultString="SDH - "+confidence+"% confident"
elif(inferredResult[2] == 1):
	confidence = format(finalPercentages[2], '.4f')
	resultString="ICH - "+ confidence +"% confident"
elif(inferredResult[3] == 1):
	confidence = format(finalPercentages[3], '.4f')
	resultString="IVH - "+cofidence+"% confident"
elif(inferredResult[4] == 1):
	confidence = format(finalPercentages[4], '.4f')
	resultString="NO HEMORRHAGE - "+confidence+"% confident"
else:
	print("something")
resultingImage = cv2.cvtColor(contourTestImage, cv2.COLOR_GRAY2BGR)
cv2.drawContours(resultingImage, [testingContour], 0, (0,255,0), 5)
cv2.putText(resultingImage, resultString, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), True)
cv2.namedWindow("Final Contour", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Final Contour", resultingImage)
cv2.waitKey()
cv2.destroyAllWindows()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
