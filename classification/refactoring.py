import matplotlib.pyplot as plt
import skimage.feature as skimg
import skimage.measure as skm
import numpy as np
import cv2
import glob
import os

from scipy.stats import kurtosis, skew
from pathlib import Path

#global definitions
in_neurons = 18
out_neurons = 3
samplesize = 251	#arbitrarily chosen
learningRate = -0.0075 
testingContour = None
contourTestImage = None
grayscaleImagesList = None
contourImagesList = None
epochNumber = 10000


#---------------------------------------------------------------------------
#defining parameters for a single-hidden-layer neural network
hl_neurons = 18
w0_1 = np.zeros((in_neurons, hl_neurons))
w1_1 = np.zeros((hl_neurons, out_neurons))
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#defining parameters for a 2-hidden-layer neural network
l1_neurons = 12
l2_neurons = 6
w0_2 = np.zeros((in_neurons, l1_neurons))
w1_2 = np.zeros((l1_neurons, l2_neurons))
w2_2 = np.zeros((l2_neurons, out_neurons))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


testNumber = 1
params = str(testNumber)+": learning rate = "+str(learningRate)+"; number of nodes in 1HL NN: "+str(hl_neurons)+"\n"

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
    filename = "Training/trainingInputMatrix.txt"
    pathname = Path(filename)

    if pathname.is_file():
        inputMatrix = np.loadtxt(filename)
        print("read training input matrix from file")

    else:
        pathname_grayscale = "Training/grayscale/*.png" # refine path name!!!
        filenames_grayscale = sorted(glob.glob(pathname_grayscale))
        pathname_contour = "Training/contours/*.png" #change path name!!!
        outfile = open("filenames_gs.txt", 'w')
        outfile.write("\n".join(filenames_grayscale))
        outfile.close()
        filenames_contour = sorted(glob.glob(pathname_contour))

        inputMatrix = np.empty((samplesize, in_neurons))
        print("generating input training matrix. . .")
        for i in range(0, len(filenames_grayscale)):
            currentImagePath = filenames_grayscale[i]
            image = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
            grayscale = getGrayscaleFeatures(image)
            currentImagePath = filenames_contour[i]
            image_contour = cv2.imread(currentImagePath, cv2.IMREAD_GRAYSCALE)
            contours = getShapeFeatures(image_contour)
            inputMatrix[i, 0:10] = grayscale
            inputMatrix[i, 10:18] = contours
        print("done")
        np.savetxt(filename, inputMatrix)

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
    filename = "Training/trainingExpectedOutputMatrix.txt"
    pathname = Path(filename)

    if pathname.is_file():
        expectedOutputMatrix = np.loadtxt(filename)
        print("read training expected output matrix from file")

    else:
        print("generating training expectd output matrix . . .")
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
        np.savetxt(filename, expectedOutputMatrix)
        print("done")
    return expectedOutputMatrix
#-----------------------s---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#train the neural network
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x) 

def initialiseNeuralNetwork():
    global w0_1
    global w1_1
    global w0_2
    global w1_2
    global w2_2
    
    np.random.seed(1)
    w0_2 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
    w1_2 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
    w2_2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1
    w0_1 = 2 * np.random.random((in_neurons, hl_neurons)) - 1
    w1_1 = 2 * np.random.random((hl_neurons, out_neurons)) - 1

def TrainNeuralNetwork(inputMatrix, outputMatrix):
	
    X = inputMatrix
    Y = outputMatrix
    global w0_2
    global w1_2
    global w2_2
    global w0_1
    global w1_1
    
    #create new path for new folder where everything will be stored
    resultsFolder = "Training/Tests/"+str(testNumber)+"/"
    os.makedirs(resultsFolder)

    #open file to add details re training
    testList = open("Training/Tests/testKey.txt", "a+")
    testList.write(params)
    testList.close()
    
    #text files storing MSE recorded for each NN
    mse_1 = open("Training/Tests/"+str(testNumber)+"/1_MSE.txt", 'w') 
    mse_2 = open("Training/Tests/"+str(testNumber)+"/2_MSE.txt", 'w')
    diff_1 = open("Training/Tests/"+str(testNumber)+"/1_Difference.txt", 'w')
    diff_2 = open("Training/Tests/"+str(testNumber)+"/2_Difference.txt", 'w')
    miscInfo = open("Training/Tests/"+str(testNumber)+"/OtherValues.txt", 'w') 
    i = 0
    initialiseNeuralNetwork()
    totalError_1 = 10.0
    totalError_2 = 10.0

    nn1_converged = False
    nn2_converged = False
    #while (totalMeanError_1hl > 0.00005) and (totalMeanError_2hl > 0.00005) :

    for i in range(0, epochNumber):
        prevEpochError_1 = totalError_1
        prevEpochError_2 = totalError_2
        total_dW_in_1_1 = 0
        total_dW_1_out_1 = 0
        totalError_1 = 0

        total_dW_in_1_2 = 0
        total_dW_1_2_2 = 0
        total_dW_2_out_2 = 0
        totalError_2 = 0

        for j in range (0, samplesize):
            l0 = np.array(X[j,], ndmin=2)
            np.reshape(l0, (1, in_neurons))

            ############## ONE HIDDEN LAYER NEURAL NETWORK STRUCTURE 
            ###### feedforward
            if(nn1_converged == False):
                s1_1 = np.dot(l0, w0_1)#working out hidden layer 
                z1_1 = sigmoid(s1_1)
                z1_1_deriv = sigmoid_derivative(z1_1).T
                s2_1 = np.dot(z1_1, w1_1) #woroing out outputs
                z2_1 = sigmoid(s2_1)
                yHat_1 = z2_1   #computed output

                ###### mean square error
                totalError_1 += np.mean(np.sum(0.5 * (np.square(Y[j, : ] - yHat_1)))) #<-- total error of neural network
                #print(str(totalError))

                ###### back propagation
                delta_out_1 = (yHat_1 - Y[j,:]).T
                delta_1_1 = np.multiply(z1_1_deriv, np.dot(w1_1, delta_out_1))

                ###### calculate change in weights needed and update weights
                dW_in_1_1 = learningRate * (np.dot(delta_1_1, l0).T)
                total_dW_in_1_1  += np.mean(dW_in_1_1)

                dW_1_out_1 = learningRate * (np.dot(delta_out_1, z1_1).T)
                total_dW_1_out_1 += np.mean(dW_1_out_1)
                w0_1 += dW_in_1_1
                w1_1 += dW_1_out_1
                #end of 1 hidden layer

            if(nn2_converged == False):
                ############## TWO HIDDEN LAYER NEURAL NETWORK STRUCTURE
                ###### feedforward
                s1_2 = np.dot(l0, w0_2)
                z1_2 = sigmoid(s1_2)
                z1_2_deriv = sigmoid_derivative(z1_2).T
                s2_2 = np.dot(z1_2, w1_2)
                z2_2 = sigmoid(s2_2)
                z2_2_deriv = sigmoid_derivative(z2_2).T
                s3_2 = np.dot(z2_2, w2_2)
                z3_2 = sigmoid(s3_2)
                yHat_2 = z3_2
            
                ###### mean square error
                totalError_2 += np.mean(np.sum(0.5 * (np.square(Y[j, : ] - yHat_2))))

                ###### back propagation
                delta_out_2 = (yHat_2 - Y[j,:]).T
                delta2_2 = np.multiply(z2_2_deriv, np.dot(w2_2, delta_out_2))
                delta1_2 = np.multiply(z1_2_deriv, np.dot(w1_2, delta2_2))

                ###### calculate change in weights needed and update weights
                dW_in_1_2 = learningRate * (np.dot(delta1_2, l0).T)
                total_dW_in_1_2 += np.mean(dW_in_1_2)
                dW_1_2_2 = learningRate * (np.dot(delta2_2, z1_2).T)
                total_dW_1_2_2 += np.mean(dW_1_2_2)
                dW_2_out_2 = learningRate * (np.dot(delta_out_2, z2_2).T)
                total_dW_2_out_2 += np.mean(dW_2_out_2)
                w0_2 += dW_in_1_2
                w1_2 += dW_1_2_2
                w2_2 += dW_2_out_2
     
        total_dW_in_1_1 /=  float(samplesize)
        total_dW_1_out_1 /= float(samplesize)
        totalError_1 /= float(samplesize)
        
        total_dW_in_1_2 /= float(samplesize)
        total_dW_1_2_2 /= float(samplesize)
        total_dW_2_out_2 /= float(samplesize)
        totalError_2 /= float(samplesize)

        mse_1.write(str(totalError_1)+"\n")
        diff_1.write(str(prevEpochError_1 - totalError_1)+"\n")
        mse_2.write(str(totalError_2)+"\n")
        diff_2.write(str(prevEpochError_2 - totalError_2)+"\n")

        print("1HL: EPOCH "+str(i)+": "+str(totalError_1)+"|||"+str(float(total_dW_in_1_1))+"|"+str(float(total_dW_1_out_1)))
        print("2HL: EPOCH "+str(i)+": "+str(totalError_2)+"|||"+str(float(total_dW_in_1_2))+"|"+str(float(total_dW_1_2_2))+"|"+str(float(total_dW_2_out_2)))
        
        if(abs(prevEpochError_1 - totalError_1) < 0.000001 and (nn1_converged == False)):
            nn1_converged = True
            miscInfo.write("[1HL]   converged at epoch "+str(i+1))
            miscInfo.write("[1HL]   final difference: "+str(prevEpochError_1 - totalError_1))
            
        if(abs(prevEpochError_2 - totalError_2) < 0.000001 and (nn2_converged == False)):
            nn2_converged = True
            print(str(prevEpochError_2 - totalError_2))
            miscInfo.write("[2HL]   converged at epoch "+str(i+1))
            miscInfo.write("[2HL]   final difference: "+str(prevEpochError_2 - totalError_2))
        
        

        if(nn1_converged and nn2_converged):
            #store weights after training network and include details about test in log file being kept
            np.savetxt("Training/Tests/"+str(testNumber)+"/1_"+str(testNumber)+"_w0.txt", w0_1)
            np.savetxt("Training/Tests/"+str(testNumber)+"/1_"+str(testNumber)+"_w1.txt", w1_1)
            np.savetxt("Training/Tests/"+str(testNumber)+"/2_"+str(testNumber)+"_w0.txt", w0_2)
            np.savetxt("Training/Tests/"+str(testNumber)+"/2_"+str(testNumber)+"_w1.txt", w1_2)
            np.savetxt("Training/Tests/"+str(testNumber)+"/2_"+str(testNumber)+"_w2.txt", w2_2)
            
            break

    mse_1.close()
    mse_2.close()
    diff_1.close()
    diff_2.close()
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
    global w0_2
    global w1_2
    global w2_2
    global w0_1
    global w1_1

    resultMatrix_1hl = np.empty((len(inputMatrix), out_neurons))
    resultMatrix_2hl = np.empty((len(inputMatrix), out_neurons))
    for i in range(0, len(inputMatrix)):
        l0 = np.array(inputMatrix[i,], ndmin=2)
        np.reshape(l0, (1, in_neurons))
        l1 = sigmoid(np.dot(l0, w0_2))
        l2 = sigmoid(np.dot(l1, w1_2))
        l3 = sigmoid(np.dot(l2, w2_2))
    
        l1_1 = sigmoid(np.dot(l0, w0_1))
        l2_1 = sigmoid(np.dot(l1_1, w1_1))
        
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
normalisedTrainingInputMatrix = normaliseMatrix(trainingInputMatrix)
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
results_2hl = processResults(finalProbabilities2hl)
outfile2 = open("results 2hl.txt", 'w')
outfile2.write("\n".join(results_2hl))
outfile2.close()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
