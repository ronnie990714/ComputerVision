import cv2
import numpy as np
import matplotlib.pyplot as plt

def arr_mul(numbers, num):
    return list(map(lambda x: x * num, numbers))

arr = [1, 2, 3, 2, 1]
img_path = './img/test.jpg'

if __name__== '__main__':
    img_file = cv2.imread(img_path)
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    sobel = [arr_mul(arr, -1), arr_mul(arr, -2) , arr_mul(arr, 0), arr_mul(arr, 1), arr_mul(arr, 2)]
    kernel_sobel = np.array(sobel)

    # Original image
    plt.subplot(2, 3, 1)
    plt.title('original')
    plt.axis('off')
    plt.imshow(img)

    # average filter image
    plt.subplot(2, 3, 2)
    blured_avg = cv2.boxFilter(img, -1, (5, 5), normalize=True)
    plt.title('avg filter')
    plt.axis('off')
    plt.imshow(blured_avg)

    # sobel filter image
    plt.subplot(2, 3, 3)
    blured_sobel = cv2.filter2D(img, -1, kernel_sobel)
    plt.title('sobel filter')
    plt.axis('off')
    plt.imshow(blured_sobel)

    # Gaussian blur image
    for i in range(3):
        plt.subplot(2, 3, 4 + i)
        blured_gauss = cv2.GaussianBlur(img, (5, 5), 2 * i + 1)
        plt.title("Gaussian sigma = {}".format(2 * i + 1))
        plt.axis('off')
        plt.imshow(blured_gauss)

    plt.show()
