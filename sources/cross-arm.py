import cv2 as cv
import numpy as np 
import os 
import math

click_count = 0
top_x = 0
top_y = 0
bot_x = 0
bot_y = 0

def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global click_count, top_x, top_y, bot_x, bot_y
        if click_count == 0:
            top_x = x
            top_y = y
            print("First ROI", top_x, top_y)
            click_count += 1
        elif click_count == 1:
            bot_x = x
            bot_y = y
            print("Second ROI", bot_x, bot_y)
            click_count = 0
    
def main():
    print("Arm Detection")
    print(cv.__version__)
    # alg = cv.GeneralizedHough()
    # print(algo.read())
    # ballard = cv.GeneralizedHoughBallard()
    img_number = 53
    img_max = 59
    img_min = 53
    template = np.zeros((5,5))
    # alg.setTemplate(template)
    start_detect = False
    while True:
        filename = '\\images\\images'+str(img_number)+'.png'
        file_location = os.getcwd() + filename
        # print("File Location : ", file_location)
        rgb_img = cv.imread(file_location)
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        # sobelx = cv.Sobel(blur_img, cv.CV_64F, 1, 0, ksize=5)
        # sobely = cv.Sobel(blur_img, cv.CV_64F, 0, 1, ksize=5)
        # sobel = sobelx + sobely
        # laplacian = cv.Laplacian(blur_img, cv.CV_64F)
        canny = cv.Canny(blur_img, 0, 150)
        # print("canny shape", rgb_img.shape)
        # cv.imshow('RGB Image', rgb_img)
        # lines = cv.HoughLines(canny, 1, np.pi / 90, 150, None, 0, 0)
    
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv.line(rgb_img, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        if start_detect:
            # print(type(alg))
            position, votes = alg.detect(canny)
            # print("Result ", alg.read())
            
        # cv.imshow('RGB Image', rgb_img)
        cv.imshow('Gray Image', canny)
        cv.imshow('Template', template)
        cv.setMouseCallback('Gray Image', click_event)
        key = cv.waitKey(1)
        if key == ord('x'):
            break
        elif key == ord('n'):
            img_number += 1
            if img_number > img_max:
                img_number = img_min
        elif key == ord('p'):
            img_number -= 1
            if img_number < img_min:
                img_number = img_max
        # Save template
        elif key == ord('t'):
            template = canny[top_y:bot_y,top_x:bot_x]
            row, col = template.shape
            cx = col // 2
            cy = row // 2
            print("Center ", cx, cy)
            gh = cv.GeneralizedHough()
            print(type(gh))
            gh.setTemplate(template)
            # alg.setTemplate(template)
        # Start matching
        elif key == ord('d'):
            start_detect = True

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()