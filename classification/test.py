import matplotlib.pyplot as plt
import skimage.feature as skimg
import skimage.measure as skm
import numpy as np
import cv2
import glob

from scipy.stats import kurtosis, skew

#global definitions
in_neurons = 18
out_neurons = 3
samplesize = 251	#arbitrarily chosen
learningRate = -0.005 
testingContour = None
contourTestImage = None
grayscaleImagesList = None
contourImagesList = None
epochNumber = 100


#---------------------------------------------------------------------------
#defining parameters for a single-hidden-layer neural network
hl_neurons = 18
w0_1hl = np.zeros((in_neurons, hl_neurons))
w1_1hl = np.zeros((hl_neurons, out_neurons))
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#defining parameters for a 2-hidden-layer neural network
l1_neurons = 12
l2_neurons = 6
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

    #print("in getGrayFeatures")
    for i in range(1, len(areas)):
    	areas[i] = cv2.contourArea(contours[i])
    	if(areas[i] > maxArea):
    		maxArea = areas[i]
    		maxAreaLoc = i


    maxPerim = cv2.arcLength(contours[maxAreaLoc], True)
    #print(str(maxArea)+" || AT || "+str(maxAreaLoc)+" || PERIM = "+str(maxPerim))
    drawing = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    global testingContour
    testingContour = contours[maxAreaLoc]
    cv2.drawContours(drawing, contours, maxAreaLoc, (0,255,0), 5)
    #cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("contours", drawing)
    #cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
    #cv2.imshow("original", originalImage)
    #cv2.waitKey()

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
	outfile = open("filenames_gs.txt", 'w')
	outfile.write("\n".join(filenames_grayscale))
	outfile.close()
	filenames_contour = sorted(glob.glob(pathname_contour))
	#outfile2 = open("filenames_cont.txt", 'w')
	#outfile2.write("\n".join(filenames_contour))
	#np.savetxt("filenames_contours.txt", filenames_contour)
	#np.savetxt("filenames_grayscale.txt", filenames_grayscale)
	inputMatrix = np.empty((samplesize, in_neurons))
	print("generating input training matrix. . .")
	for i in range(0, len(filenames_grayscale)):
		#print(str(i+1))
		currentImagePath = filenames_grayscale[i]
		image = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
		#plt.figure(i+1)
		#plt.imshow(image, cmap='gray', interpolation='bicubic')

		grayscale = getGrayscaleFeatures(image)

		currentImagePath = filenames_contour[i]
		image_contour = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
		contours = getShapeFeatures(image_contour)
		#plt.figure(i+len(filenames_grayscale)+1)
		#plt.imshow(image_contour, cmap='gray', interpolation='bicubic')
    
		inputMatrix[i, 0:10] = grayscale
		inputMatrix[i, 10:18] = contours
	print("done")
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
	trainingExpectedResults = open("Training/expected_outputs.txt", "r")
	index = 0
	for line in trainingExpectedResults:
		if(line == "EDH\n"):
			expectedOutputMatrix[index] = [1,0,0]#,0,0]
			#print("epidural")
		elif( line == "SDH\n"):
			expectedOutputMatrix[index] = [0,1,0]#,0,0]
			#print ("subdural")
		elif( line == "ICH\n"):
			expectedOutputMatrix[index] = [0,0,1]#,0,0]
			#print ("intracranial")
		#elif( line == "IVH\n"):
		#	expectedOutputMatrix[index] = [0,0,0,1,0]
		#	print ("intra-ventricular")
		#elif( line == "NO\n"):
		#	expectedOutputMatrix[index] = [0,0,0,0,1]
		#	print ("no hemorrhage detected")
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
	return x * (1 - x) 

def initialiseNeuralNetwork():
    global w0
    global w1
    global w2
    global w0_1hl
    global w1_1hl
    
    np.random.seed(1)
    w0 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
    w1 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
    w2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1
    w0_1hl = 2 * np.random.random((in_neurons, hl_neurons)) - 1
    w1_1hl = 2 * np.random.random((hl_neurons, out_neurons)) - 1
   
def TrainNeuralNetwork(inputMatrix, outputMatrix):
	#np.random.seed(1)
    X = inputMatrix
    Y = outputMatrix
    global w0
    global w1
    global w2
    global w0_1hl
    global w1_1hl
    
    error_1 = open("error_1_hidden_layer.txt", 'w')
    error_2 = open("error_2_hidden_layer.txt", 'w')
    totalMeanError_1hl = 100.0
    totalMeanError_2hl = 100.0
    i = 0
    initialiseNeuralNetwork()
    print(w0_1hl)
    print(w1_1hl)
    #while (totalMeanError_1hl > 0.00005) and (totalMeanError_2hl > 0.00005) :
    
    for i in range(0, epochNumber):
        total_dW_in_1 = 0
        total_dW_1_out = 0
        for j in range (0, samplesize):
            l0 = np.array(X[j,], ndmin=2)
            np.reshape(l0, (1, in_neurons))

            #1 layer 
            #++++feedforward
            s1 = np.dot(l0, w0_1hl)
            z1 = sigmoid(s1)
            z1_deriv = sigmoid_derivative(z1).T
            
            s2 = np.dot(z1, w1_1hl)
            z2 = sigmoid(s2)
            
            yHat = z2
            #total error = summation{1/2 * (target - output)^2}
            x = Y[j, : ] - yHat
            x2 = np.square(x)
            x3 = 0.5 * x2
            totalError = np.sum(x3) #<-- total error of neural network
            #print(str(totalError))
            delta_out = (yHat - Y[j,:]).T
            temp1 = np.dot(w1_1hl, delta_out)
            delta_1 = np.multiply(z1_deriv, temp1)

            tempDW_1 = np.dot(delta_1, l0).T
            deltaW_in_1 = learningRate * tempDW_1
            total_dW_in_1  = total_dW_in_1 + np.mean(deltaW_in_1)

            deltaW_1_out = learningRate * (np.dot(delta_out, z1).T)
            total_dW_1_out = total_dW_1_out + np.mean(deltaW_1_out)
            w0_1hl += deltaW_in_1
            w1_1hl += deltaW_1_out
        
        total_dW_in_1 = total_dW_in_1 / float(epochNumber)
        total_dW_1_out = total_dW_1_out / float(epochNumber)
        print("EPOCH "+str(i)+": "+str(float(total_dW_in_1))+"|"+str(float(total_dW_1_out)))
        total_dW_in_1 = 0;
        total_dW_1_out = 0;


    error_1.close()
    error_2.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#get test image from file and test input matrix
def getTestInputMatrix():
	#possible imprivement - get image to be tested  for using glob?
    global contourTestImage
    global grayscaleImagesList
    global contourImagesList
    pathname_contour_test = "Testing/contours/*.png" # refine path name!!!
    filenames_contour_test = sorted(glob.glob(pathname_contour_test))
    pathname_grayscale_test = "Testing/grayscale/*.png" # refine path name!!!
    filenames_grayscale_test = sorted(glob.glob(pathname_grayscale_test))

    grayscaleImagesList = filenames_grayscale_test
    contourImagesList = filenames_contour_test

    testInput = np.empty((len(filenames_contour_test), in_neurons))
    
    for i in range(0, len(filenames_contour_test)):
        currentTestImage = filenames_grayscale_test[i]
        image = cv2.imread(currentTestImage, cv2.IMREAD_GRAYSCALE)
        grayscaleTest = getGrayscaleFeatures(image) 

        currentTestImage = filenames_contour_test[i]
        image = cv2.imread(currentTestImage, cv2.IMREAD_GRAYSCALE)
        contoursTest = getShapeFeatures(image)
        testInput[i, 0:10] = grayscaleTest
        testInput[i, 10:18] = contoursTest

    return testInput
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pass test case through neural network
def TestNeuralNetwork(inputMatrix):
    global w0
    global w1
    global w2
    global w0_1hl
    global w1_1hl

    resultMatrix_1hl = np.empty((len(inputMatrix), out_neurons))
    resultMatrix_2hl = np.empty((len(inputMatrix), out_neurons))
    for i in range(0, len(inputMatrix)):
        l0 = np.array(inputMatrix[i,], ndmin=2)
        np.reshape(l0, (1, in_neurons))
        l1 = sigmoid(np.dot(l0, w0))
        l2 = sigmoid(np.dot(l1, w1))
        l3 = sigmoid(np.dot(l2, w2))
    
        l1_1 = sigmoid(np.dot(l0, w0_1hl))
        l2_1 = sigmoid(np.dot(l1_1, w1_1hl))
        
        resultMatrix_1hl[i, :] = l2_1
        resultMatrix_2hl[i, :] = l3

        print("for image "+str(i))
        print(l2_1)
        print(l3)
    
    return resultMatrix_1hl, resultMatrix_2hl
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def processResults(resultMatrix):
    global grayscaleImagesList
    global contourImagesList
    percentages = np.multiply(resultMatrix, 100)
    results = []
    inferredResults = np.zeros((len(resultMatrix), out_neurons))
    for i in range (0, len(resultMatrix)):
        maxProbability = 0
        maxProbabilityIndex = -1
        index = 0
        for value in percentages[i,]:
            if value > maxProbability:
                maxProbability = value
                maxProbabilityIndex = index
            index += 1
        inferredResults[i, maxProbabilityIndex] = 1

        resultString = ""
        if(inferredResults[i, 0] == 1):
            confidence = format(percentages[i, 0], '.4f')
            resultString = "Case "+str(i + 1)+": EDH - "+ confidence +"% confident"
        if(inferredResults[i, 1] == 1):
            confidence = format(percentages[i, 1], '.4f')
            resultString = "Case "+str(i + 1)+": SDH - "+ confidence +"% confident"
        if(inferredResults[i, 2] == 1):
            confidence = format(percentages[i, 2], '.4f')
            resultString = "Case "+str(i + 1)+": ICH - "+ confidence +"% confident"
        
        results.append(resultString)
        currentImagePath_gs = grayscaleImagesList[i]
        currentImage_gs = cv2.imread(currentImagePath_gs, cv2.IMREAD_GRAYSCALE)

        currentImagePath_contour = contourImagesList[i]
        currentImage_contour = cv2.imread(currentImagePath_contour, cv2.IMREAD_GRAYSCALE)

        resultImage = getDrawnContourImage(currentImage_gs, currentImage_contour)

        cv2.putText(resultImage, resultString, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), True)
        windowName = "Final Result for case "+str(i+1)
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windowName, resultImage)
        cv2.waitKey()
        cv2.destroyWindow(windowName)
    
    return results

def getDrawnContourImage(gs_image, contour_image):
    #original = np.copy(image)
    im, contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = np.zeros([len(contours)])
    maxArea = 0
    maxAreaLoc = -1

    for i in range(1, len(areas)):
    	areas[i] = cv2.contourArea(contours[i])
    	if(areas[i] > maxArea):
    		maxArea = areas[i]
    		maxAreaLoc = i

    #im in following line needs to be the grayscale image
    drawing = cv2.cvtColor(gs_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(drawing, contours, maxAreaLoc, (0,255,0), 5)

    return drawing


trainingInputMatrix = getTrainingInputMatrix()
#normalisedTrainingInputMatrix = normaliseMatrix(trainingInputMatrix)
trainingOutputMatrix = getTrainingOutputMatrix()
TrainNeuralNetwork(trainingInputMatrix, trainingOutputMatrix)
testInputMatrix = getTestInputMatrix()
finalProbabilities1hl, finalProbabilities2hl = TestNeuralNetwork(testInputMatrix)

#For the single hidden layer neural network
results_1hl = processResults(finalProbabilities1hl)
outfile = open("results 1hl.txt", 'w')
outfile.write("\n".join(results_1hl))
outfile.close()

#For the two hidden layer neural network
#results_2hl = processResults(finalProbabilities2hl)
#outfile2 = open("results 2hl.txt", 'w')
#outfile2.write("\n".join(results_2hl))
#outfile2.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
