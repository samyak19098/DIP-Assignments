import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import *


#-----Common utility function------#

def roundNumber(n):

	diff_floor = n - floor(n)
	diff_ceil = ceil(n) - n
	if (diff_floor >= diff_ceil):
		return int(ceil(n))
	return int(floor(n))
#----------------------------------#


#-------------------------------------Question 3 Starts----------------------------------------------------------#

def computeNormalizedHistogramQ3(img):

	M = len(img)
	N = len(img[0])
	maxPixelValue = 255

	p_value = [0] * (maxPixelValue + 1)
	for k in range(maxPixelValue + 1):
		occurrences = np.where(img == k)
		count = len(occurrences[0])
		p_k = count / (M * N)
		p_value[k] = p_k

	imageNormalizedHistogram = p_value

	return imageNormalizedHistogram


def computeCDFQ3(normalizedHistogram, maxPixelValue):

	cdf = [0] * (maxPixelValue + 1)
	cdf[0] = normalizedHistogram[0]
	for k in range(1, maxPixelValue + 1):
		cdf[k] = cdf[k - 1] + normalizedHistogram[k]

	return cdf


def histogramEqualizeQ3(inputImage):

	M = len(inputImage)
	N = len(inputImage[0])
	maxPixelValue = 255

	inputNormalizedHistogram = computeNormalizedHistogramQ3(inputImage)

	cdf = computeCDFQ3(inputNormalizedHistogram, maxPixelValue)

	mappedPixels = {}
	for k in range(maxPixelValue + 1):
		mappedPixels[k] = roundNumber(maxPixelValue * cdf[k])
	equalizedImage = inputImage.copy()
	for k in range(maxPixelValue + 1):
		coord = np.where(inputImage == k)
		equalizedImage[coord] = mappedPixels[k]

	return equalizedImage




def displayImageAndHistoQ3(inputImage, outputImage, h, g):

	#display the images
	inputImage = inputImage.astype(np.uint8)
	outputImage = outputImage.astype(np.uint8)
	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Output Image", outputImage)


	#display the plots
	pixelList = list(range(0, 256))
	plt.figure(1)
	plt.bar(pixelList, h)
	plt.xlabel("r")
	plt.ylabel("p(r)")
	plt.title("Histogram for Input Image")
	# plt.show()
	plt.figure(2)
	plt.bar(pixelList, g)
	plt.xlabel("r")
	plt.ylabel("p(r)")
	plt.title("Histogram for Equalized Image")
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def question3():

	inputImage = cv2.imread('./cameraman.jpg', 0)
	equalizedImage = histogramEqualizeQ3(inputImage)

	#computing the histograms to display the same
	inputNormalizedHistogram = computeNormalizedHistogramQ3(inputImage)
	equalizedImgNormalizedHistogram = computeNormalizedHistogramQ3(equalizedImage)
	print("::::: Input Image Normalized Histogram :::::")
	print()
	print(inputNormalizedHistogram)
	print()
	print(":::::: Output (equalized) image Normalized Histogram :::::")
	print()
	print(equalizedImgNormalizedHistogram)
	displayImageAndHistoQ3(inputImage, equalizedImage, inputNormalizedHistogram, equalizedImgNormalizedHistogram)

#----------------------------------Question 3 Ends here----------------------------------------------------------#



#-------------------------------------Question 4 Starts----------------------------------------------------------#

def gammaTransformQ4(inputImage, gamma):

	maxValue = np.amax(inputImage)
	c = maxValue / (maxValue ** gamma)

	M = len(inputImage)
	N = len(inputImage[0])
	outputImage = np.zeros((M, N))

	maxPixelValue = 255
	mapping = {}
	for i in range(maxPixelValue + 1):
		transformedPixelValue = roundNumber(c * (i ** gamma))
		mapping[i] = transformedPixelValue
		coord = np.where(inputImage == i)
		outputImage[coord] = transformedPixelValue

	return outputImage


def computeNormalizedHistogramQ4(img):

	M = len(img)
	N = len(img[0])
	maxPixelValue = 255

	p_value = [0] * (maxPixelValue + 1)
	for k in range(maxPixelValue + 1):
		occurrences = np.where(img == k)
		count = len(occurrences[0])
		p_k = count / (M * N)
		p_value[k] = p_k

	imageNormalizedHistogram = p_value

	return imageNormalizedHistogram


def computeCDFQ4(normalizedHistogram, maxPixelValue):

	cdf = [0] * (maxPixelValue + 1)
	cdf[0] = normalizedHistogram[0]
	for k in range(1, maxPixelValue + 1):
		cdf[k] = cdf[k - 1] + normalizedHistogram[k]

	return cdf


def histogramMatchingQ4(inputImage, targetImage):

	M = len(inputImage)
	N = len(inputImage[0])
	maxPixelValue = 255

	inputNormalizedHisto_h = computeNormalizedHistogramQ4(inputImage)
	targetNormalizedHisto_g = computeNormalizedHistogramQ4(targetImage)

	inputCDF_H = computeCDFQ4(inputNormalizedHisto_h, maxPixelValue)
	targetCDF_G = computeCDFQ4(targetNormalizedHisto_g, maxPixelValue)

	mappedPixel = {}
	for k in range(maxPixelValue + 1):
		abs_diff = np.absolute(np.array(targetCDF_G) - inputCDF_H[k])
		s_value = np.argmin(abs_diff)
		mappedPixel[k] = s_value
	
	matchedImage = inputImage.copy()
	for k in range(maxPixelValue + 1):
		coord = np.where(inputImage == k)
		matchedImage[coord] = mappedPixel[k]

	return matchedImage

def displayImageAndHistoQ4(inputImage, targetImage, matchedImage, h, g, f):

	#display images
	inputImage = inputImage.astype(np.uint8)
	targetImage = targetImage.astype(np.uint8)
	matchedImage = matchedImage.astype(np.uint8)
	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Target Image", targetImage)
	cv2.imshow("Matched Image", matchedImage)

	#plot histograms
	pixelList = list(range(0, 256))
	plt.figure(1)
	plt.bar(pixelList, h)
	plt.xlabel("r")
	plt.ylabel("p(r)")
	plt.title("Histogram for Input Image")
	# plt.show()
	plt.figure(2)
	plt.bar(pixelList, g)
	plt.xlabel("r")
	plt.ylabel("p(r)")
	plt.title("Histogram for Target Image")
	# plt.show()

	plt.figure(3)
	plt.bar(pixelList, f)
	plt.xlabel("r")
	plt.ylabel("p(r)")
	plt.title("Histogram for Matched Image")
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def question4():

	inputImage = cv2.imread('./cameraman.jpg', 0)
	print("Press 'i' to give a gamma input or 'd' to run default(0.5)?")
	option = input("Response(i/d): ")
	if(option == "i"):
		gammaValue = float(input("gamma : "))
	else:
		gammaValue = 0.5

	targetImage = gammaTransformQ4(inputImage, gammaValue)
	matchedImage = histogramMatchingQ4(inputImage, targetImage)

	matchedNormalizedHistogram = computeNormalizedHistogramQ4(matchedImage)
	inputNormalizedHistogram = computeNormalizedHistogramQ4(inputImage)
	targetNormalizedHistogram = computeNormalizedHistogramQ4(targetImage)

	print("::::: Input Image Normalized Histogram :::::")
	print()
	print(inputNormalizedHistogram)
	print()
	print("::::: Target Image Normalized Histogram :::::")
	print()
	print(targetNormalizedHistogram)
	print()
	print("::::: Matched Image Normalized Histogram :::::")
	print()
	print(matchedNormalizedHistogram)
	print()

	displayImageAndHistoQ4(inputImage, targetImage, matchedImage, inputNormalizedHistogram, targetNormalizedHistogram, matchedNormalizedHistogram)

#----------------------------------Question 4 Ends here----------------------------------------------------------#



#-------------------------------------Question 5 Starts----------------------------------------------------------#
def rotateFilter(filter):

	rotatedFilter = np.rot90(filter, 2)
	return rotatedFilter


def computeOverlap(paddedInputImage, rotatedFilter, ranges):
	# ranges = [left, right, upper, down]
	value = 0
	l, r, u, d = ranges[0], ranges[1], ranges[2], ranges[3]
	# print(rotatedFilter)
	# print(paddedInputImage)
	for i in range(u, d + 1):
		for j in range(l, r + 1):
			# print(i, j, i - u, j - l, " : ", paddedInputImage[i][j] * rotatedFilter[i - u][j - l])
			value = value + (paddedInputImage[i][j] * rotatedFilter[i - u][j - l])
	return value


def performConvolution(inputImage, filter):

	M = len(inputImage)
	N = len(inputImage[0])
	P = len(filter)
	Q = len(filter[0])

	rotatedFilter = rotateFilter(np.array(filter))

	print(":::::: Original Filter :::::::\n", filter, "\n")
	print(":::::: Rotated Filter :::::::\n", rotatedFilter, "\n")

	paddedInputImage = np.pad(inputImage, ((2, 2), (2, 2)), 'constant')

	outputRows = M + P - 1
	outputCols = N + Q - 1

	output = np.zeros((outputRows + 2, outputCols + 2), dtype=int)

	for i in range(1, outputRows + 1):
		for j in range(1, outputCols + 1):
			rangeList = [(j - 1), (j + 1), (i - 1), (i + 1)]
			output[i][j] = computeOverlap(paddedInputImage, rotatedFilter, rangeList)

	# output = output.astype(np.uint32)
	return output


def generateRandMatrix(n, lowerLimit, upperLimit):

	matrix = np.random.randint(lowerLimit, upperLimit + 1, size=(n, n))
	return matrix


def cropMatrix(matrix, cropRows, cropCols):

	matrix = np.delete(matrix, cropRows, axis=0)
	matrix = np.delete(matrix, cropCols, axis=1)
	return matrix


def question5():


	print("Enter 'R' for choosing random matrices, 'I' for giving input, 'D' to run on default case given in the assignment.")
	option = input("Response (R/I/D) : ")
	if(option == 'R'):
		print("Enter a valid lower and upper limit (b/w 0 to 255) for choosing random elements")

		lowerLimit = int(input("Lower Limit: "))
		upperLimit = int(input("Upper Limit: "))
		inputMatrix = generateRandMatrix(3, lowerLimit, upperLimit)
		filterMatrix = generateRandMatrix(3, lowerLimit, upperLimit)

	elif(option == 'I'):
		print("---Enter the Input Matrix---")
		inputMatrix = np.ones((3,3)) 
		for i in range(3):
			for j in range(3):
				inputMatrix[i][j] = int(input())
		print("------------------")
		print()
		print("---Enter the Filter Matrix---")
		filterMatrix = np.ones((3, 3))
		for i in range(3):
			for j in range(3):
				filterMatrix[i][j] = int(input())
		print("------------------")
		print()
	else:
		inputMatrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
		filterMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

	print("##--------- RESULTS --------------#")
	print("::::: Input Matrix :::::\n", inputMatrix)
	print("NOTE : Origin for input corresponds to position : (0, 0) in the input matrix")
	print()
	convOut = performConvolution(inputMatrix, filterMatrix)
	finalConvolutedOutput = cropMatrix(convOut, (0, 6), (0, 6))
	print("::::: Output Matrix :::::\n", finalConvolutedOutput)
	print("NOTE : Origin for output corresponds to position : (1, 1) in the output matrix")
#----------------------------------Question 5 Ends here----------------------------------------------------------#

if __name__ == "__main__":

	#for checking question 3 just call the function question3() (uncomment the function call)
	# question3()
	#for checking question 4 just call the function question3() (uncomment the function call)
	# question4()
	#for checking question 5 just call the function question3() (uncomment the function call)
	question5()
