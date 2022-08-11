import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def main():
    print("Template Matching")
    main_image = cv.imread('TestData/img1.jpg', 0)
    # main_image_copy = main_image.copy()
    template = cv.imread('TestData/img1t2.jpg', 0)
    template_w, template_h = template.shape[::-1]

    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for method in methods:
        img = main_image.copy()

        result = cv.matchTemplate(img, template, eval(method))
        # print(method, ": ", result)   # debug
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)

        if eval(method) in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = minLoc
        else:
            top_left = maxLoc

        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        cv.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(result, cmap='gray')
        plt.title('matching result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)
        plt.show()

if __name__ == "__main__":
    main()
