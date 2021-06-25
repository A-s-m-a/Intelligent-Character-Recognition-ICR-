import cv2
from tensorflow.keras.models import load_model
import numpy as np


def preprocessing(input_image, edge=False, inv_thresh=False):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if inv_thresh:
        ret, im_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im_th = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
        im_th = cv2.bitwise_not(im_th)
    else:
        ret, im_th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_th = cv2.adaptiveThreshold(im_th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    if edge:
        edge_image = cv2.Canny(im_th, 0, 255)
        return edge_image
    return im_th


CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

img = cv2.imread(r'data\test1.jpg')
# cv2.imshow("img", img)
# cv2.waitKey(0)
edged = preprocessing(img, edge=True, inv_thresh=True)
# cv2.imshow('edged', edged)
# cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rects = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in contours]
rects = [cv2.boundingRect(ctr) for ctr in contours]
# [print(index, rect) for index, rect in enumerate(rects)]
# print(len(rects))
result = ''
print(result)
model = load_model('weights.h5')
for index, rect in enumerate(rects):
    # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 1)
    # length = int(rect[3] * 1.6)
    # pt1 = int(rect[1] + rect[3] // 2 - length // 2)
    # pt2 = int(rect[0] + rect[2] // 2 - length // 2)
    # roi = img[pt1:pt1 + length, pt2:pt2 + length]
    roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    roi = preprocessing(roi)
    # cv2.imshow('ROI', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(index)

    # cv2.imwrite(f'.\\ROI\\{index}.jpg', roi)

    image = cv2.resize(roi, (28, 28))
    image = image.reshape(-1, 28, 28, 1).astype(float)
    prediction = np.argmax(model.predict(image), axis=-1)
    cv2.putText(img, CATEGORIES[prediction[0]], (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
    result += CATEGORIES[prediction[0]]

# cv2.imshow('contours', img)
# cv2.waitKey(0)
cv2.imwrite('contoured1.jpg', img)
# print(result)
