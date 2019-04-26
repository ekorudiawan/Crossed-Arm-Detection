import numpy as np
import cv2 as cv
import scipy.spatial.distance as dist

cap = cv.VideoCapture('../videos/bob1-left.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(1000,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
height, width, _ = old_frame.shape
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Ambil point yg kanan saja
p0 = p0[p0[:,:,1] > (height//2)].reshape(-1,1,2)
p_init = p0.copy()
# print(p_init.shape)

index_num = np.argwhere(p0[:,:,0] > (width//2))
list_index_num = list(index_num[:,0])
# print(list_index_num)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
frame_num = 0
req_frame = 100

while True:
    ret, frame = cap.read()
    if ret:
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Draw boundary lines
        frame = cv.line(frame, (0,height//2),(width,height//2), (255,0,0), 2)
        frame = cv.line(frame, (width//2,height),(width//2,0), (255,255,0), 2)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        lost_idx = np.argwhere(st == 0)
        # print("Lost Index : ", lost_idx)
        # print("Lest los :", len(lost_idx))
        if len(lost_idx) > 0:
            for i in range(len(lost_idx)):
                delete_num = int(lost_idx[i,0])
                # print("Len pinit ", len(p_init))
                p_init = np.delete(p_init, delete_num, axis=0)
                # print("Len pinit ", len(p_init))
                # Update list index
                p_init = p_init[p_init[:,:,1] > (height//2)].reshape(-1,1,2)
                index_num = np.argwhere(p_init[:,:,0] > (width//2))
                list_index_num = list(index_num[:,0])
                # print(list_index_num)

            # list_index_num.remove(lost_idx[:,0])
            # print("Hapus dulu")
        # print("Loss ", lost_idx)
        # print("Index ", list_index_num)
        # print(p0.shape)
        for i in range(len(good_new)):
            color = (0,0,0)
            if i in list_index_num:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            mask = cv.line(mask, (good_old[i,0],good_old[i,1]),(good_new[i,0],good_new[i,1]), color, 2)
            frame = cv.circle(frame,(good_old[i,0],good_old[i,1]),5, color, -1)
        img = cv.add(frame,mask)
        # if frame_num == 0:
        #     good_init = good_old.copy()
        # print("Frame Number : ", frame_num)
        # print("St ", st)
        # # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #     frame = cv.circle(frame,(a,b),5, color[i].tolist(), -1)
        # img = cv.add(frame,mask)

        # # Detection
        # if frame_num > req_frame:
        #     for i,(new,old) in enumerate(zip(good_new,good_init)):
        #         a,b = new.ravel()
        #         c,d = old.ravel()
        #         mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #         frame = cv.circle(frame,(a,b),5, color[i].tolist(), -1)
        #     img = cv.add(frame,mask)
        # # else:
        # #     img = frame.copy()

        cv.imshow('frame',img)
        k = cv.waitKey(100) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        frame_num += 1
    elif frame_num > 2:
        break
    else:
        break
cv.destroyAllWindows()
cap.release()