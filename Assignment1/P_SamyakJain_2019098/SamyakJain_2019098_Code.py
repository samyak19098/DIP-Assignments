import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import *


#-------------------------------------Question 3 Starts-------------------------------------------------#

def displayImageQ3(inputImage, outputImage):
	inputImage = inputImage.astype(np.uint8)
	outputImage = outputImage.astype(np.uint8)
	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Output Image", outputImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def bilinearInterpolation(img, c):
	''' c -> interpolation factor
		img -> input image
	'''
	m1, n1 = len(img), len(img[0])
	m2 = int(m1 * c)
	n2 = int(n1 * c)

	#doing padding on input image by mirroring technique
	inputImage = np.zeros((m1 + 2, n1 + 2))

	for i in range(m1):
		for j in range(n1):
			inputImage[i][j] = img[i][j]
	for i in range(m1):
		for j in range(n1, len(inputImage[0])):
			inputImage[i][j] = 0
	for i in range(m1, len(inputImage)):
		for j in range(len(inputImage[0])):
			inputImage[i][j] = 0

	outputImage = np.ones((m2, n2))*-1

	roundingConstant = 10**(-6)
	for i in range(m2):
		for j in range(n2):
			if(outputImage[i][j] == -1):
				x_inp = i / c
				y_inp = j / c

				lx = round(x_inp + roundingConstant)
				rx = round(((i + c) / c) + roundingConstant)
				uy = round(y_inp + roundingConstant)
				dy = round(((j + c) / c) + roundingConstant)

				lx = int(lx)
				rx = int(rx)
				uy = int(uy)
				dy = int(dy)

				X = [
						[lx, uy, (lx * uy), 1],
						[lx, dy, (lx * dy), 1],
						[rx, uy, (rx * uy), 1],
						[rx, dy, (rx * dy), 1],

					]
				V = [
						[inputImage[lx][uy]],
						[inputImage[lx][dy]],
						[inputImage[rx][uy]],
						[inputImage[rx][dy]],
					]

				if(np.linalg.det(X) == 0):
					#singular, add delta*I
					delta = 0.00001
					X = np.add(X, (np.identity(4)*delta))

				A = np.dot(np.linalg.inv(X), V)

				outputExpression = np.array([x_inp, y_inp, (x_inp * y_inp), 1])
				outputPixelValue = np.dot(outputExpression, A)[0]
				outputImage[i][j] = round(outputPixelValue)
	return outputImage

def question3():
	img = cv2.imread("./SamyakJain_2019098_original512.bmp", 0)
	interpolationFactor = float(input("Interpolation Factor : "))
	output = bilinearInterpolation(img, interpolationFactor)
	displayImageQ3(img, output)

#----------------------------------Question 3 Ends here----------------------------------------------------------#



#--------------------------------------Question 4 Starts----------------------------------------------------------------#

def displayImageQ4(inputImage, outputImage):
	inputImage = inputImage.astype(np.uint8)
	outputImage = outputImage.astype(np.uint8)
	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Output Image", outputImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def geometricTransform(img, transformMatrix):

	m1, n1 = len(img), len(img[0])
	m2 = (8 * m1) + 1
	n2 = (8 * n1) + 1

	#creating a 0-padded input image
	inputImage = np.zeros((m1 + 4, n1 + 4))
	for i in range(2, m1 + 2):
		for j in range(2, n1 + 2):
			inputImage[i][j] = img[i - 2][j - 2]

	#output grid
	outputImage = np.ones((m2, n2))*-1

	roundingConstant = 10**(-6)
	for i in range(m2):
		for j in range(n2):

			x_prime = i - (4*m1)
			y_prime = j - (4*n1)
			coordInp = np.dot(np.array([x_prime, y_prime, 1]), np.linalg.inv(transformMatrix))
			
			x_inp = coordInp[0]
			y_inp = coordInp[1]

			if(x_inp < 0 or y_inp < 0 or x_inp >= m1 or y_inp >= n1):
				outputImage[i][j] = 0

			else:

				if( (floor(x_inp) == ceil(x_inp)) and (floor(y_inp) == ceil(y_inp)) ):
					outputImage[i][j] = inputImage[int(x_inp) + 2][int(y_inp) + 2]
				else:
					lx = int(floor(x_inp))
					rx = int(ceil(x_inp))
					dy = int(floor(y_inp))
					uy = int(ceil(y_inp))
					X = [
							[lx, dy, (lx * dy), 1],
							[lx, uy, (lx * uy), 1],
							[rx, dy, (rx * dy), 1],
							[rx, uy, (rx * uy), 1],
						]
					V = [
							[inputImage[lx + 2][dy + 2]],
							[inputImage[lx + 2][uy + 2]],
							[inputImage[rx + 2][dy + 2]],
							[inputImage[rx + 2][uy + 2]],
						]
					if(np.linalg.det(X) == 0):
						#singular, add delta*I
						delta = 0.00001
						X = np.add(X, (np.identity(4)*delta))

					A = np.dot(np.linalg.inv(X), V)

					outputExpression = np.array([x_inp, y_inp, (x_inp * y_inp), 1])
					outputPixelValue = np.dot(outputExpression, A)[0]
					outputImage[i][j] = int(round(outputPixelValue))
	return outputImage
def calcCos(theta):
	return np.cos(theta*np.pi/180)
def calcSin(theta):
	return np.sin(theta*np.pi/180)

def constructTransformMatrix(transforms):
	matrix = transforms[0]
	for i in range(1, len(transforms)):
		matrix = np.dot(matrix, transforms[i])
	return matrix

def question4():
	img = cv2.imread("./SamyakJain_2019098_original64.jpg", 0)
	transforms = []
	print("Press 1 for input T matrix and 2 for constructing")
	option = int(input())
	if(option == 2):
		print("Press S for scaling, T for transformation and R for rotation")
		for i in range(3):
			print("Do you want to perform add a transform (y/n) ?")
			ans = input()
			if(ans == "y"):
				op = input("Operation : ")
				if(op == "S"):
					x_scaling = float(input("X-Scaling factor : "))
					y_scaling = float(input("Y-Scaling factor : "))
					scaling = np.array([[x_scaling, 0, 0], [0, y_scaling, 0], [0, 0, 1]])
					transforms.append(scaling)
				elif(op == "T"):
					x_translation = float(input("X-Translation(in px): "))
					y_translation = float(input("Y-Translation(in px): "))
					trans = np.array([[1, 0, 0], [0, 1, 0], [x_translation, y_translation, 1]])
					transforms.append(trans)
				elif(op == "R"):
					theta = float(input("Rotation(deg): "))
					rot = np.array([[calcCos(theta), -1*calcSin(theta), 0], [calcSin(theta), calcCos(theta), 0], [0, 0, 1]])
					transforms.append(rot)
			else:
				break
		T = constructTransformMatrix(transforms)
	else:
		T = np.ones((3,3)) 
		for i in range(3):
			for j in range(3):
				T[i][j] = float(input())
	print(":::::Transformation Matrix:::::")
	print(T)
	output = geometricTransform(img, T)
	displayImageQ4(img, output)


#-------------------------------------Question 4 Ends------------------------------------------------------------------#



#-------------------------------------Question 5 Start------------------------------------------------------------------#

def displayImageQ5(registeredImg, unregisteredImg, referenceImg):
	registeredImg = registeredImg.astype(np.uint8)
	unregisteredImg = unregisteredImg.astype(np.uint8)
	referenceImg = referenceImg.astype(np.uint8)
	cv2.imshow("Registered Image", registeredImg)
	cv2.imshow("Unregistered Image", unregisteredImg)
	cv2.imshow("Reference Image", referenceImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def transformAndRegisterImg(img, transformMatrix):

	m1, n1 = len(img), len(img[0])
	m2 = (2 * m1) + 1
	n2 = (2 * n1) + 1

	#creating a 0-padded input image
	inputImage = np.zeros((m1 + 4, n1 + 4))
	for i in range(2, m1 + 2):
		for j in range(2, n1 + 2):
			inputImage[i][j] = img[i - 2][j - 2]

	#output grid
	outputImage = np.ones((m2, n2))*-1

	for i in range(m2):
		for j in range(n2):

			x_prime = i - (1*m1)
			y_prime = j - (1*n1)
			coordInp = np.dot(np.array([x_prime, y_prime, 1]), np.linalg.inv(transformMatrix))
			
			x_inp = coordInp[0]
			y_inp = coordInp[1]

			if(x_inp < 0 or y_inp < 0 or x_inp >= m1 or y_inp >= n1):
				outputImage[i][j] = 0

			else:

				if( (floor(x_inp) == ceil(x_inp)) and (floor(y_inp) == ceil(y_inp)) ):
					outputImage[i][j] = inputImage[int(x_inp) + 2][int(y_inp) + 2]
				else:
					lx = int(floor(x_inp))
					rx = int(ceil(x_inp))
					dy = int(floor(y_inp))
					uy = int(ceil(y_inp))
					X = [
							[lx, dy, (lx * dy), 1],
							[lx, uy, (lx * uy), 1],
							[rx, dy, (rx * dy), 1],
							[rx, uy, (rx * uy), 1],
						]
					V = [
							[inputImage[lx + 2][dy + 2]],
							[inputImage[lx + 2][uy + 2]],
							[inputImage[rx + 2][dy + 2]],
							[inputImage[rx + 2][uy + 2]],
						]
					if(np.linalg.det(X) == 0):
						#singular, add delta*I
						lambdaFactor = 0.00001
						X = np.add(X, (np.identity(4)*lambdaFactor))

					A = np.dot(np.linalg.inv(X), V)

					outputExpression = np.array([x_inp, y_inp, (x_inp * y_inp), 1])
					outputPixelValue = np.dot(outputExpression, A)[0]
					outputImage[i][j] = int(round(outputPixelValue))
	return outputImage

def imageRegistration(unregImg, pointsUnreg, pointsRef):

	Z_inv = np.dot(np.linalg.inv(pointsUnreg), pointsRef)
	Z = np.linalg.inv(Z_inv)
	print(":::: Z :::::")
	print(Z)
	registeredImg = transformAndRegisterImg(unregImg, Z_inv)

	return registeredImg

def question5():
	unregImg = cv2.imread("./SamyakJain_2019098_unreg.jpg", 0)
	refImg = cv2.imread("./SamyakJain_2019098_original64.jpg", 0)
	unregPoints = np.array([[180, 6, 1]
						,[210 ,31 , 1]
						,[191 ,28 ,1]])

	refPoints = np.array([[61.5, 44.6, 1]
						,[63.3, 64, 1]
						,[57.6, 56.2, 1]])

	registeredImg = imageRegistration(unregImg, unregPoints, refPoints)
	displayImageQ5(registeredImg, unregImg, refImg)

#--------------------------------------Question 5 Ends------------------------------------------------------------------#


#----------------------------------Assignment Demo Starts----------------------#


if __name__ == "__main__":

	#for checking question 3 just call the function question3() (uncomment the function call)
	# question3()

	#for checking question 4 just call the function question4() (uncomment the function call)
	# question4()

	#for checking question 5 just call the function question5() (uncomment the function call)
	# question5()

#---------------------------------Assignment Demo Ends-------------------------#