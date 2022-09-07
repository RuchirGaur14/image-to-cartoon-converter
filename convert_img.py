import imghdr
import cv2
import numpy as np

def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detect(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

def color_quantasation(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, centre = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centre = np.uint8(centre)
    result = centre[label.flatten()]
    result = result.reshape(img.shape)
    return result

print("Enter picture name")
b = input()
a = ("./" + b + ".jpg")

img = read_img(a)
line_wdt = 9
blur_value = 7
total_colors = 20

edge_img = edge_detect(img, line_wdt, blur_value)
img = color_quantasation(img, total_colors)
blur_img = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
cartoon = cv2.bitwise_and(blur_img, blur_img, mask = edge_img)
cv2.imwrite("cartoon_" + b + ".jpg", cartoon)