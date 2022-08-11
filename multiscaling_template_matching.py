import cv2 as cv
from matplotlib import pyplot as plt
import copy
import imutils

def main():
    print("Multiscaling Template Matching")
    main_img = cv.imread("TestData/img1.jpg")
    resize_image = main_img.copy()
    template = cv.imread("TestData/img1t1.jpg")
    template_w, template_h = template.shape[:-1]
    result_dict = {}

    method = 'cv.TM_SQDIFF_NORMED'
    scaling_factor = 1.15   # factor by which the original image will we be scaled
    limiting_factor = 2 # factor indicating max resizing of the original image

    while resize_image.shape[0] < main_img.shape[0] * limiting_factor and \
            resize_image.shape[1] < main_img.shape[1] * limiting_factor:
        result = cv.matchTemplate(resize_image, template, eval(method))
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
        result_dict[resize_image.shape] = [minVal, maxVal, minLoc, maxLoc]
        resize_image = imutils.resize(resize_image, width=int(resize_image.shape[1] * scaling_factor),
                                      height=resize_image.shape[0],
                                      inter=cv.INTER_CUBIC)

        # debug code

        # main_img_copy = main_img.copy() # debug
        #
        # if eval(method) in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        #     top_left = minLoc
        # else:
        #     top_left = maxLoc
        #
        # bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        #
        # cv.rectangle(main_img_copy, top_left, bottom_right, 255, 2)
        #
        # plt.subplot(121), plt.imshow(result, cmap='gray')
        # plt.title('matching result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(main_img_copy, cmap='gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle("Name")
        # plt.show()

    for key in result_dict:   # debug
        print(key)

    if eval(method) in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        key = find_minVal(result_dict)
        top_left = result_dict[key][2]
    else:
        key = find_maxVal(result_dict)
        top_left = result_dict[key][3]

    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    mainImgResize = main_img.copy()
    mainImgResize = imutils.resize(mainImgResize, width=key[1], height=key[0],
                                   inter=cv.INTER_CUBIC)

    cv.rectangle(mainImgResize, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(main_img, cmap='gray')
    plt.title('matching result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(mainImgResize, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle("Name")
    plt.show()


def find_minVal(dict):
    minKey = list(dict.keys())[0]
    for key in dict:
        if dict[key][0] < dict[minKey][0]:
            minKey = key

    return minKey

def find_maxVal(dict):
    maxKey = list(dict.keys())[0]
    for key in dict:
        if dict[key][1] > dict[maxKey][1]:
            maxKey = key

    return maxKey

if __name__=="__main__":
    main()
    # resize_Image()