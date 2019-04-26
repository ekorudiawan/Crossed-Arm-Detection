import cv2 as cv
import numpy as np 
import os 
import math
from collections import defaultdict

click_count = 0

temp_tx = 0
temp_ty = 0
temp_bx = 0
temp_by = 0
mouse_x = 0
mouse_y = 0

first_click = False
second_click = False

MIN_CANNY_THRESHOLD = 0
MAX_CANNY_THRESHOLD = 5

def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global click_count, temp_tx, temp_ty, temp_bx, temp_by
        global first_click, second_click
        if click_count == 0:
            temp_tx = x
            temp_ty = y
            print("First ROI", temp_tx, temp_ty)
            click_count += 1
            first_click = True
            second_click = False
        elif click_count == 1:
            temp_bx = x
            temp_by = y
            print("Second ROI", temp_bx, temp_by)
            click_count = 0
            first_click = False
            second_click = True
    elif event == cv.EVENT_MOUSEMOVE:
        global mouse_x, mouse_y
        # print("Mouse", mouse_x, mouse_y)
        mouse_x = x
        mouse_y = y

def main():
    print("Arm Detection")
    print(cv.__version__)
    img_number = 1
    img_max = 626
    img_min = 1

    # Load template image from file
    rtop_temp_img = cv.imread('./temp_img/right_top.png', 0)
    ltop_temp_img = cv.imread('./temp_img/left_top.png', 0)

    alg = cv.createGeneralizedHoughBallard()
    # alg = cv.createGeneralizedHoughGuil()
    start_detect = True

    r_temp_w = 0
    r_temp_h = 0

    l_temp_w = 0
    l_temp_h = 0

    l_temp_h, l_temp_w = ltop_temp_img.shape
    r_temp_h, r_temp_w = rtop_temp_img.shape
    # Get Guil Parameter
    # print("Angle Epsilon : ", alg.getAngleEpsilon())
    # print("Angle Step : ", alg.getAngleStep())
    # print("Angle Thresh : ", alg.getAngleThresh())
    # print("Levels : ", alg.getLevels())
    # print("Max Angles : ", alg.getMaxAngle())
    # print("Max Scales : ", alg.getMaxScale())
    # print("Min Angles : ", alg.getMinAngle())
    # print("Min Scales : ", alg.getMinScale())
    # print("Pos Thresh : ", alg.getPosThresh())
    # print("Scale Step : ", alg.getScaleStep())
    # print("Scale Thresh : ", alg.getScaleThresh())
    # print("Xi : ", alg.getXi())

    # Get Ballard Parameter
    # print("Angle Epsilon : ", alg.getAngleEpsilon())
    # print("Angle Step : ", alg.getAngleStep())
    # print("Angle Thresh : ", alg.getAngleThresh())
    print("Levels : ", alg.getLevels())
    print("Votes Threshold : ", alg.getVotesThreshold())
    print("Minimum Distance : ", alg.getMinDist())
    print("DP : ", alg.getDp())
    print("Max Buffer : ", alg.getMaxBufferSize())
    print("Votes Threshold : ", alg.getVotesThreshold())

    # print("Max Angles : ", alg.getMaxAngle())
    # print("Max Scales : ", alg.getMaxScale())
    # print("Min Angles : ", alg.getMinAngle())
    # print("Min Scales : ", alg.getMinScale())
    # print("Pos Thresh : ", alg.getPosThresh())
    # print("Scale Step : ", alg.getScaleStep())
    # print("Scale Thresh : ", alg.getScaleThresh())
    # print("Xi : ", alg.getXi())

    # Set Guil Parameter
    # alg.setAngleEpsilon()
    # alg.setAngleStep(30)
    # alg.setAngleThresh()
    # alg.setLevels()
    # alg.setMaxAngle(90)
    # alg.setMaxScale(1.5)
    # alg.setMinAngle()
    # alg.setMinScale(0.5)
    # alg.setPosThresh()
    # alg.setScaleStep(0.5)
    # alg.setScaleThresh()
    # alg.setXi()

    # Set Ballard Parameter
    alg.setLevels(360)
    alg.setVotesThreshold(30)
    alg.setMinDist(100)
    alg.setDp(2)
    alg.setMaxBufferSize(1000)

    print("Setup Done")
    while True:

        filename = '/home/images/images ('+str(img_number)+').png'

        rgb_img = cv.imread(filename)
        rgb_img = cv.resize(rgb_img,(640, 480), interpolation = cv.INTER_CUBIC)
         
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
        # rgb_img = cv.filter2D(rgb_img, -1, kernel_sharpening)
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        # blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        # blur_img = cv.bilateralFilter(gray_img,9,50,50)

        # v = np.median(blur_img)
        # sigma = 0.5
        # lower_thresh = int(max(0, (1.0 - sigma) * v))
        # upper_thresh = int(min(255, (1.0 + sigma) * v))
        # MIN_CANNY_THRESHOLD = lower_thresh
        # MAX_CANNY_THRESHOLD = upper_thresh
        # canny = cv.Canny(blur_img, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD)
        rgb_canny = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

        # Draw ROI
        global mouse_x, mouse_y
        global first_click, second_click
        if first_click:
            rgb_canny = cv.rectangle(rgb_canny, (temp_tx,temp_ty), (mouse_x,mouse_y), (0,0,127), 2)
        elif second_click:
            rgb_canny = cv.rectangle(rgb_canny, (temp_tx,temp_ty), (temp_bx,temp_by), (0,0,127), 2)

        if start_detect:
            left_detected = False
            right_detected = False
            detection_stage = ['right', 'left']
            left_votes = 0
            right_votes = 0
            for detection in detection_stage:
                if detection == 'right':
                    alg.setTemplate(rtop_temp_img)
                elif detection == 'left':
                    alg.setTemplate(ltop_temp_img)

                positions, votes = alg.detect(gray_img)
                print("Pos ", detection, positions)
                print("Votes ", detection, votes)
                if positions is not None:
                    if detection == 'right':
                        right_votes = votes[0][0][0]
                    elif detection == 'left':
                        left_votes = votes[0][0][0]
                    posx = positions[0][0][0]
                    posy = positions[0][0][1]
                    scale = positions[0][0][2]
                    angle = positions[0][0][3]

                    if detection == 'right':
                        print("Right Temp Pos :", posx, posy)
                    elif detection == 'left':
                        print("Left Temp Pos :", posx, posy)

                    print("Scale :", scale)
                    print("Angle :", angle)
                    if detection == 'right' :
                        roi_tx = int(posx - (r_temp_w//2))
                        roi_ty = int(posy - (r_temp_h//2))
                        roi_bx = int(posx + (r_temp_w//2))
                        roi_by = int(posy + (r_temp_h//2))
                    elif detection == 'left':
                        roi_tx = int(posx - (l_temp_w//2))
                        roi_ty = int(posy - (l_temp_h//2))
                        roi_bx = int(posx + (l_temp_w//2))
                        roi_by = int(posy + (l_temp_h//2))

                    detected_color = (0, 0, 0)

                    if detection == 'right':
                        right_detected = True
                        detected_color = (255, 0, 0)
                    elif detection == 'left':
                        left_detected = True
                        detected_color = (0, 255, 0)

                    rgb_img = cv.rectangle(rgb_img, (roi_tx,roi_ty), (roi_bx,roi_by), detected_color, 3)
                    
            if right_votes > left_votes:
                rgb_img = cv.putText(rgb_img,'Right Hand on Top', (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
                print("Right Hand on Top")
            elif left_votes and right_votes:
                rgb_img = cv.putText(rgb_img,'Left Hand on Top', (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
                print("Left Hand on Top")
            else:
                print("Unable to Recognize, No One Detected")
            
        cv.imshow('Gray Image', rgb_canny)
        cv.imshow('RGB Image', rgb_img)
        cv.imshow('Right Top Temp', rtop_temp_img)
        cv.imshow('Left Top Temp', ltop_temp_img)
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
        # Press right to set template
        elif key == ord('r'):
            first_click = False
            second_click = False
            rtop_temp_img = gray_img[temp_ty:temp_by, temp_tx:temp_bx]
            r_temp_h, r_temp_w = rtop_temp_img.shape
            cv.imwrite("./temp_img/right_top.png", rtop_temp_img)
        elif key == ord('l'):
            first_click = False
            second_click = False
            ltop_temp_img = gray_img[temp_ty:temp_by, temp_tx:temp_bx]
            l_temp_h, l_temp_w = ltop_temp_img.shape
            cv.imwrite("./temp_img/left_top.png", ltop_temp_img)
        # Start matching
        elif key == ord('d'):
            start_detect = True

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()