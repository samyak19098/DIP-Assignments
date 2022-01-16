import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import *


#-------------------------------------Question 1 Starts--------------------------------------------------------------------#

def display_Q1(original_img, noised_img, output_img, cls, lap):
    original_img = original_img.astype(np.uint8)
    noised_img = noised_img.astype(np.uint8)
    output_img = output_img.astype(np.uint8)
    cv2.imshow("Original Image", original_img)
    cv2.imshow("Noised Image", noised_img)
    cv2.imshow("Output Image", output_img)

    plt.figure(1)
    plt.imshow(cls, cmap='gray')
    plt.title("CLS FILTER on best lambda")

    plt.figure(2)
    plt.imshow(lap, cmap='gray')
    plt.title("LAPLACIAN FILTER")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def computeMSE(orig, f_hat):
    squared_diff_matrix = np.square(np.subtract(orig, f_hat))
    mse_value = squared_diff_matrix.sum() / (squared_diff_matrix.shape[0] * squared_diff_matrix.shape[1])
    return mse_value


def computePSNR(orig, f_hat):
    mse_val = computeMSE(orig, f_hat)
    return 10 * np.log10((255 * 255) / mse_val)


def perform_Q1(noised_img, original_img):

    m, n = original_img.shape
    padded_original_img = np.zeros((2 * m, 2 * n))
    padded_noised_img = np.zeros((2 * m, 2 * n))
    for i in range(m):
        for j in range(n):
            padded_original_img[i][j] = original_img[i][j]
            padded_noised_img[i][j] = noised_img[i][j]
    lambda_values = np.arange(0, 1.25, 0.25).tolist()
    print(f'lambda values : {lambda_values}')

    h_box = np.ones((11, 11)) / (11 * 11)
    padded_h_box = np.zeros((2 * m, 2 * n))
    for i in range(h_box.shape[0]):
        for j in range(h_box.shape[1]):
            padded_h_box[i][j] = h_box[i][j]
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    padded_laplacian = np.zeros((2 * m, 2 * n))
    for i in range(laplacian.shape[0]):
        for j in range(laplacian.shape[1]):
            padded_laplacian[i][j] = laplacian[i][j]

    # computing ffts
    G = np.fft.fft2(padded_noised_img)
    H = np.fft.fft2(padded_h_box)
    L = np.fft.fft2(padded_laplacian)

    min_mse = 99999999999
    best_lambda = -1

    for lambda_val in lambda_values:
        print(f'\nfor lambda = {lambda_val}:')
        cls_filter = np.divide(np.conjugate(H), (np.square(np.abs(H)) + (lambda_val * np.square(np.abs(L)))))
        F_hat = np.multiply(cls_filter, G)
        f_hat = np.real(np.fft.ifft2(F_hat))[:m, :n]
        curr_mse_value = computeMSE(original_img, f_hat)
        if curr_mse_value < min_mse:
            min_mse = curr_mse_value
            best_lambda = lambda_val
        PSNR_value = computePSNR(original_img, f_hat)
        print(f'MSE: {curr_mse_value} | PSNR = {PSNR_value}\n--------------------------')

    # best_restored_image
    cls_filter_best = np.divide(np.conjugate(H), (np.square(np.abs(H)) + (best_lambda * np.square(np.abs(L)))))
    cls_filter_plt = np.log10(1 + np.abs(np.fft.fftshift(cls_filter_best)))
    laplacian_filter_plt = np.log10(1 + np.abs(np.fft.fftshift(L)))

    F_hat_best = np.multiply(cls_filter_best, G)
    f_hat_best = np.real(np.fft.ifft2(F_hat_best))[:m, :n]
    mse_best = computeMSE(original_img, f_hat_best)
    PSNR_best = computePSNR(original_img, f_hat_best)
    print(f'For best restored image : \nLambda = {best_lambda}\nMSE Value = {mse_best}\nPSNR Value = {PSNR_best}\n')

    return f_hat_best, best_lambda, cls_filter_plt, laplacian_filter_plt


def question_1():
    original_image = cv2.imread('./cameraman.jpg', 0)
    noised_image = cv2.imread('./noiseIm.jpg', 0)
    best_restored_img, best_lambda_value, cls_filter_best, laplacian_filter_plt = perform_Q1(noised_image, original_image)
    display_Q1(original_image, noised_image, best_restored_img, cls_filter_best, laplacian_filter_plt)

#-------------------------------------Question 1 Ends--------------------------------------------------------------------#


#-------------------------------------Question 3 Starts--------------------------------------------------------------------#

def perform_equalization(inputImage):
    m, n = inputImage.shape
    totalPoints = m * n
    maxPixelValue = 255
    inputNormalizedHistogram = np.zeros(maxPixelValue + 1)
    for k in range(maxPixelValue + 1):
        inputNormalizedHistogram[k] = np.count_nonzero(inputImage == k) / (totalPoints)
    cdf = [0] * (maxPixelValue + 1)
    cdf[0] = inputNormalizedHistogram[0]
    for k in range(1, maxPixelValue + 1):
        cdf[k] = cdf[k - 1] + inputNormalizedHistogram[k]

    mappedPixels = [0] * (maxPixelValue + 1)
    for k in range(maxPixelValue + 1):
        mappedPixels[k] = round(maxPixelValue * cdf[k])

    equalizedImage = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            orig = inputImage[i][j]
            equalizedImage[i][j] = mappedPixels[orig]
    return equalizedImage, inputNormalizedHistogram

def display_Q3(input_image, hsi_image, hsi_image_equalized, rgb_image, rgb_image_equalized, h1, h2):
    cv2.imshow("Input Image", input_image)
    cv2.imshow("HSI Image", hsi_image)
    cv2.imshow("HSI Equalized", hsi_image_equalized)
    cv2.imshow("RGB Image", rgb_image)
    cv2.imshow("RGB Equalized", rgb_image_equalized)

    pixelList = list(range(0, 256))
    plt.figure(1)
    plt.bar(pixelList, h1)
    plt.xlabel("r")
    plt.ylabel("p(r)")
    plt.title("Intensity Channel Histogram (Input Image)")
    # plt.show()
    plt.figure(2)
    plt.bar(pixelList, h2)
    plt.xlabel("r")
    plt.ylabel("p(r)")
    plt.title("Intensity Channel Histogram (Equalized Image)")
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_component_1(comp, r, g, b):
    mat = np.zeros((512, 512))
    if (comp == "H"):
        for i in range(512):
            for j in range(512):
                numerator = ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j])) / 2
                denominator = (((r[i][j] - g[i][j]) ** 2) + ((r[i][j] - b[i][j]) * (g[i][j] - b[i][j]))) ** 0.5
                if (denominator == 0):
                    denominator = denominator + 0.0001
                theta = acos(numerator / denominator)
                if (b[i][j] <= g[i][j]):
                    mat[i][j] = theta
                else:
                    mat[i][j] = (2 * np.pi) - theta
    elif (comp == "S"):
        for i in range(512):
            for j in range(512):
                denom = (r[i][j] + g[i][j] + b[i][j])
                if (denom == 0):
                    denom = denom + 0.0001
                mat[i][j] = 1 - ((3 / denom) * min(min(r[i][j], g[i][j]), b[i][j]))
    elif (comp == "I"):
        for i in range(512):
            for j in range(512):
                mat[i][j] = (r[i][j] + g[i][j] + b[i][j]) / 3
    return mat

def toRad(degree):
    return np.deg2rad(degree)

def get_component_2(comp, h, s, i_comp):
    mat = np.zeros((512, 512))

    if(comp == "R"):
        for i in range(512):
            for j in range(512):
                if(h[i][j] >= toRad(0) and h[i][j] < toRad(120)):
                    mat[i][j] = i_comp[i][j] * (1 + ((s[i][j] * cos(h[i][j])) / cos(toRad(60) - h[i][j])))
                elif(h[i][j] >= toRad(120) and h[i][j] < toRad(240)):
                    h_new = h[i][j] - toRad(120)
                    mat[i][j] = i_comp[i][j] * (1 - s[i][j])
                elif(h[i][j] >= toRad(240) and h[i][j] <= toRad(360)):
                    h_new = h[i][j] - toRad(240)
                    g_comp = i_comp[i][j] * (1 - s[i][j])
                    b_comp = i_comp[i][j] * (1 + ((s[i][j] * cos(h_new)) / cos(toRad(60) - h_new)))
                    mat[i][j] = (3 * i_comp[i][j]) - (g_comp + b_comp)
    elif(comp == "G"):
        for i in range(512):
            for j in range(512):
                if(h[i][j] >= toRad(0) and h[i][j] < toRad(120)):
                    r_comp = i_comp[i][j] * (1 + ((s[i][j] * cos(h[i][j])) / cos(toRad(60) - h[i][j])))
                    b_comp = i_comp[i][j] * (1 - s[i][j])
                    mat[i][j] = (3 * i_comp[i][j]) - (r_comp + b_comp)
                elif(h[i][j] >= toRad(120) and h[i][j] < toRad(240)):
                    h_new = h[i][j] - toRad(120)
                    mat[i][j] = i_comp[i][j] * (1 + ((s[i][j] * cos(h_new) / cos(toRad(60) - h_new))))
                elif(h[i][j] >= toRad(240) and h[i][j] <= toRad(360)):
                    h_new = h[i][j] - toRad(240)
                    mat[i][j] = i_comp[i][j] * (1 - s[i][j])
    elif(comp == "B"):
        for i in range(512):
            for j in range(512):
                if(h[i][j] >= toRad(0) and h[i][j] < toRad(120)):
                    mat[i][j] = i_comp[i][j] * (1 - s[i][j])
                elif(h[i][j] >= toRad(120) and h[i][j] < toRad(240)):
                    h_new = h[i][j] - toRad(120)
                    r_comp = i_comp[i][j] * (1 - s[i][j])
                    g_comp = i_comp[i][j] * (1 + ((s[i][j] * cos(h_new)) / cos(toRad(60) - h_new)))
                    mat[i][j] = (3 * i_comp[i][j]) - (r_comp + g_comp)
                elif(h[i][j] >= toRad(240) and h[i][j] <= toRad(360)):
                    h_new = h[i][j] - toRad(240)
                    mat[i][j] = i_comp[i][j] * (1 + ((s[i][j] * cos(h_new)) / cos(toRad(60) - h_new)))
    return mat

def convert_RGB_to_HSI(r_component, g_component, b_component):
    H = get_component_1("H", r_component, g_component, b_component)
    S = get_component_1("S", r_component, g_component, b_component)
    I = get_component_1("I", r_component, g_component, b_component)
    hsi_image = cv2.merge((H, S, I))
    return hsi_image, H, S, I

def convert_HSI_to_RGB(h_component, s_component, i_component):
    R = get_component_2("R", h_component, s_component, i_component)
    G = get_component_2("G", h_component, s_component, i_component)
    B = get_component_2("B", h_component, s_component, i_component)
    rgb_image = cv2.merge((R, G, B))
    return  rgb_image, R, G, B

def question_3():
    input_image_raw = cv2.imread('Fig0646(a)(lenna_original_RGB).tif', 1)
    input_image = np.float32(input_image_raw) / 255
    m, n = input_image.shape[0], input_image.shape[1]
    R_component = input_image[:, :, 0]
    G_component = input_image[:, :, 1]
    B_component = input_image[:, :, 2]
    hsi_image, H_mat, S_mat, I_mat = convert_RGB_to_HSI(R_component, G_component, B_component)
    I_mat_cp = np.int8(np.multiply(I_mat.copy(), 255))
    I_mat_eq_1, normalizedHistogram_1 = perform_equalization(I_mat_cp)
    I_mat_equalized = np.float32(I_mat_eq_1) / 255

    hsi_equalized = cv2.merge((np.float32(H_mat), np.float32(S_mat), I_mat_equalized))

    H_component_eq = hsi_equalized[:, :, 0]
    S_component_eq = hsi_equalized[:, :, 1]
    I_component_eq = hsi_equalized[:, :, 2]

    I_mat_cp_2 = np.int8(np.multiply(I_mat_equalized.copy(), 255))
    I_mat_eq_2, normalizedHistogram_2 = perform_equalization(I_mat_cp_2)

    rgb_image_conv_eq, R_mat_eq, G_mat_eq, B_mat_eq = convert_HSI_to_RGB(H_component_eq, S_component_eq, I_component_eq)
    rgb_image_conv, R_mat, G_mat, B_mat = convert_HSI_to_RGB(H_mat, S_mat, I_mat)

    display_Q3(input_image, hsi_image, hsi_equalized, rgb_image_conv, rgb_image_conv_eq, normalizedHistogram_1, normalizedHistogram_2)



#-------------------------------------Question 3 Ends--------------------------------------------------------------------#


#----------------------------------Assignment Demo Starts----------------------#


if __name__ == "__main__":

    #for checking question 1 just call the function question1() (uncomment the function call)
    # question_1()

    #for checking question 3 just call the function question1() (uncomment the function call)
    # question_3()
