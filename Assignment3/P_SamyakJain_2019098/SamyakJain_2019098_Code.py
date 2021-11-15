import numpy as np
import cv2
import matplotlib.pyplot as plt


#-------------------------------------Question 1 Starts-------------------------------------------------#

def display_Q1(inputImage, inp_zp, ms_inp_zp, cms_inp_zp, bwf, cms_bwf, outputImage, d0):
	inputImage = inputImage.astype(np.uint8)
	inp_zp = inp_zp.astype(np.uint8)
	outputImage = outputImage.astype(np.uint8)

	plt.figure(1)
	plt.imshow(ms_inp_zp, cmap='gray')
	plt.title('Mag. Spectrum Input Zero-Padded')
	plt.figure(2)
	plt.imshow(cms_inp_zp, cmap='gray')
	plt.title('Centred Mag. Spectrum Input Zero-Padded')
	plt.figure(3)
	plt.imshow(bwf, cmap='gray')
	plt.title(str("Filter at d0=" + str(d0)))
	plt.figure(4)
	plt.imshow(cms_bwf, cmap='gray')
	plt.title(str("Centred Mag. Spectrum Filter at d0=" + str(d0)))

	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Zero-padded Input Image", inp_zp)
	cv2.imshow(str("Output(d0=" + str(d0)+")"), outputImage)

	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def perform_Q1(inputImage, D0):

	N, M = inputImage.shape #N X M

	#padding the input image
	centered_padded_input_image = np.zeros((2 * N, 2 * M))
	original_zero_padded = np.zeros((2 * N, 2 * M))
	#centering the image
	for i in range(N):
		for j in range(M):
			original_zero_padded[i][j] = inputImage[i][j]
			centered_padded_input_image[i][j] = ((-1)**(i + j)) * inputImage[i][j]

	fft_original_zero_padded = np.fft.fft2(original_zero_padded)
	mag_spec_zero_padded = np.log(1 + np.abs(fft_original_zero_padded))

	#creating the filter
	butterworth_filter = np.zeros((2 * N, 2 * M))
	for u in range(2 * N):
		for v in range(2 * M):
			D_uv = np.sqrt(((u - N) ** 2) + ((v - M) ** 2))
			butterworth_filter[u][v] = 1 / (1 + (D_uv / D0)**(2*2))
	centered_mag_spec_filter = np.log(1 + np.abs(butterworth_filter))
	print(f'Filter : {butterworth_filter}\n\n' )
	#Using the filter on image
	fft_centered_padded_image = np.fft.fft2(centered_padded_input_image)
	mag_spectrum_centered_zero_padded = np.log(1 + np.abs(fft_centered_padded_image))
	element_wise_multip = np.multiply(fft_centered_padded_image, butterworth_filter)
	ifft_elem_multip = np.fft.ifft2(element_wise_multip)
	ifft_real_part = ifft_elem_multip.real

	for i in range(ifft_real_part.shape[0]):
		for j in range(ifft_real_part.shape[1]):
			ifft_real_part[i][j] = ((-1)**(i + j)) * ifft_real_part[i][j]

	#cropping the result
	cropped_output = ifft_real_part[:N, :M]

	return original_zero_padded, mag_spec_zero_padded, mag_spectrum_centered_zero_padded, butterworth_filter, centered_mag_spec_filter, cropped_output

def question1():
	inputImage = cv2.imread('./cameraman.jpg', 0)
	d0 = int(input("Enter value d0: "))
	orig_zero_pad, ms_orig_zero_pad, ms_centered_zero_pad, bw_filter, ms_centered_filter, outputResult = perform_Q1(inputImage, d0)
	display_Q1(inputImage, orig_zero_pad, ms_orig_zero_pad, ms_centered_zero_pad, bw_filter, ms_centered_filter, outputResult, d0)


#-------------------------------------Question 1 Ends------------------------------------------------------------------#

#-------------------------------------Question 3 Starts------------------------------------------------------------------#

def display_Q3(inputImage, convolution_DFT, convolution_spatial):
	inputImage = inputImage.astype(np.uint8)
	convolution_DFT = convolution_DFT.astype(np.uint8)
	convolution_spatial = convolution_spatial.astype(np.uint8)
	
	cv2.imshow("Input Image", inputImage)
	cv2.imshow("Via DFT", convolution_DFT)
	cv2.imshow("Inbuilt", convolution_spatial)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getBoxFilter(n):
	box_filter = (1 / (n * n)) * (np.ones((n, n)))
	return box_filter


def performDFTConvolution(img, box_filter):

	# # N X M input image
	N, M = img.shape
	Q, P = box_filter.shape
	zero_padded_img = np.zeros((N + Q - 1, M + P - 1))
	zero_padded_filter = np.zeros((N + Q - 1, M + P - 1))
	for i in range(N):
		for j in range(M):
			zero_padded_img[i][j] = img[i][j]
	for i in range(Q):
		for j in range(P):
			zero_padded_filter[i][j] = box_filter[i][j]

	dft_padded_img = np.fft.fft2(zero_padded_img)
	dft_padded_filter = np.fft.fft2(zero_padded_filter)
	element_wise_multip = np.multiply(dft_padded_img, dft_padded_filter)
	idft_elem_multip = np.fft.ifft2(element_wise_multip)
	idft_real_part = idft_elem_multip.real

	cropped_conv_result = idft_real_part[:N, :M]
	return cropped_conv_result

def question3():
	inputImage = cv2.imread('./cameraman.jpg', 0)
	n = int(input("Size of box-filter(n) : "))
	box_filter = getBoxFilter(n)
	print(f'--------- BOX FILTER {n}X{n} -------------\n{box_filter}\n-------------------------------\n')
	convolution_via_DFT = performDFTConvolution(inputImage, box_filter)
	spatial_convolution = cv2.filter2D(src=inputImage, ddepth=-1, kernel=box_filter)
	display_Q3(inputImage, convolution_via_DFT, spatial_convolution)

#-------------------------------------Question 3 Ends--------------------------------------------------------------------#

#-------------------------------------Question 4 Start--------------------------------------------------------------------#

def display_Q4(noisedImg, denoisedImg, input_magnitude_spectrum):
	noisedImg = noisedImg.astype(np.uint8)
	denoisedImg = denoisedImg.astype(np.uint8)

	plt.figure(1)
	plt.imshow(input_magnitude_spectrum, cmap='gray')
	plt.title('Magnitude Spectrum')
	
	cv2.imshow('Noised', noisedImg)
	cv2.imshow('Denoised', denoisedImg)
	plt.show()
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def perform_Q4(noisedImg):
	N = noisedImg.shape[0]
	M = noisedImg.shape[1]
	padded_input_image = np.zeros((2 * N, 2 * M))
	for i in range(N):
		for j in range(M):
		   padded_input_image[i][j] = ((-1)**(i + j)) * noisedImg[i][j]

	noise_filter = np.ones((2 * N, 2 * M))
	fft_noised_input = np.fft.fft2(padded_input_image)
	inputMagSpec = np.log(1 + np.abs(fft_noised_input))

	coords = [(192, 192), (320, 320)]
	print(f'Coordinates of corrupted part : {coords}\n\n')
	for i in range(coords[0][0] - 25, coords[0][0] + 26):
		for j in range(coords[0][1] - 25, coords[0][1] + 26):
			noise_filter[i][j] = 0
	for i in range(coords[1][0] - 25, coords[1][0] + 26):
		for j in range(coords[1][1] - 25, coords[1][1] + 26):
			noise_filter[i][j] = 0

	mult = np.multiply(fft_noised_input, noise_filter)
	ifft_val_real = np.fft.ifft2(mult).real

	for i in range(ifft_val_real.shape[0]):
		for j in range(ifft_val_real.shape[1]):
			ifft_val_real[i][j] = ((-1)**(i + j)) * ifft_val_real[i][j]
	cropped_img = ifft_val_real[:N, :M]
	return cropped_img, inputMagSpec

def question4():
	noisedImage = cv2.imread('./noised_img.jpg', 0)
	denoised_output, mag_spect = perform_Q4(noisedImage)
	display_Q4(noisedImage, denoised_output, mag_spect)

#-------------------------------------Question 4 Ends--------------------------------------------------------------------#


#----------------------------------Assignment Demo Starts----------------------#

if __name__ == "__main__":

	#for checking question 1 just call the function question1() (uncomment the function call)
	# question1()

	#for checking question 3 just call the function question3() (uncomment the function call)
	# question3()

	#for checking question 4 just call the function question4() (uncomment the function call)
	question4()


#---------------------------------Assignment Demo Ends-------------------------#