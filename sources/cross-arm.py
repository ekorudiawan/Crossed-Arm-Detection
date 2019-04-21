import cv2 as cv
import numpy as np 
import os 
import math
from collections import defaultdict

click_count = 0
top_x = 0
top_y = 0
bot_x = 0
bot_y = 0

MIN_CANNY_THRESHOLD = 0
MAX_CANNY_THRESHOLD = 255

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

def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    # dx = sobel(image, axis=0, mode='constant')
    # dy = sobel(image, axis=1, mode='constant')
    dx = cv.Sobel(image, cv.CV_64F, 1,0, ksize=5)
    dy = cv.Sobel(image, cv.CV_64F, 0,1, ksize=5)
    gradient = np.arctan2(dy,dx) * 180 / np.pi

    return gradient

def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    edges = cv.Canny(image, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD) 
    gradient = gradient_orientation(edges)
    
    r_table = defaultdict(list)
    for (i,j),value in np.ndenumerate(edges):
        if value:
            r_table[gradient[i,j]].append((origin[0]-i, origin[1]-j))

    return r_table

def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = cv.Canny(grayImage, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)
    
    accumulator = np.zeros(grayImage.shape)
    # print("acc shape", accumulator.shape)
    for (i,j),value in np.ndenumerate(edges):
        if value:
            for r in r_table[gradient[i,j]]:
                accum_i, accum_j = i+r[0], j+r[1]
                
                if accum_i >= 0 and accum_i < accumulator.shape[0] and accum_j >= 0 and accum_j < accumulator.shape[1]:
                    # print("accum ", accum_i, accum_j)
                    accumulator[int(accum_i), int(accum_j)] += 1
                    
    return accumulator

def general_hough_closure(reference_image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    
    Returns a function f, which takes a query image and returns the accumulator
    '''
    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2)
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image)
        
    return f

def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]

def test_general_hough(gh, reference_image, query_image):
    accumulator = gh(query_image)
    m = n_max(accumulator, 5)
    print("m", m)
    max_prob = [prob[0] for prob in m]
    y_points = [pt[1][0] for pt in m]
    x_points = [pt[1][1] for pt in m] 
    i, j = np.unravel_index(accumulator.argmax(), accumulator.shape)   
    print("max ", max_prob)
    return x_points, y_points

def main():
    print("Arm Detection")
    print(cv.__version__)
    img_number = 53
    img_max = 59
    img_min = 53
    template = np.zeros((5,5))
    start_detect = False
    detect_s = None
    while True:
        filename = '\\images\\images'+str(img_number)+'.png'
        file_location = os.getcwd() + filename
        rgb_img = cv.imread(file_location)
        gray_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
        canny = cv.Canny(blur_img, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD)

        if start_detect:
            x_points, y_points = test_general_hough(detect_s, template, canny)
            for i in range(len(x_points)):
                rgb_img = cv.circle(rgb_img,(x_points[i],y_points[i]), 50, (0,0,255), 1)
            
        cv.imshow('Gray Image', canny)
        cv.imshow('RGB Image', rgb_img)
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
            # row, col = template.shape
            # cx = col // 2
            # cy = row // 2
            # print("Center ", cx, cy)
            detect_s = general_hough_closure(template)
        # Start matching
        elif key == ord('d'):
            start_detect = True

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()