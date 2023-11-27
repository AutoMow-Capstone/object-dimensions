#################################################################################################
# References:                                                                                   #
# https://www.youtube.com/watch?v=6Otgyyv--UU                                                   #
# https://toptechboy.com/faster-launch-of-webcam-and-smoother-video-in-opencv-on-windows/       #
# https://www.youtube.com/watch?v=E46B7NPWK38&list=PLGs0VKk2DiYyXlbJVaE8y1qr24YldYNDm&index=14  #
#################################################################################################

import cv2
import numpy as np
import time

print(cv2.__version__)

width = 640
height = 360

# lower_red= np.array([160, 0, 0], dtype = "uint8")
# upper_red = np.array([255, 0, 0], dtype = "uint8")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


while True:
    # open camera
    ignore, frame = cam.read()

    # defining region of interest (lower 70% of camera frame)
    frameROI = frame[192:640,0:640]

    # convert from rgb to hsv colour model
    hsvFrame = cv2.cvtColor(frameROI, cv2.COLOR_BGR2HSV)

    # define mask 
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Morphological Transform, Dilation 
	# for each color and bitwise_and operator 
	# between imageFrame and mask determines 
	# to detect only that particular color 
    kernal = np.ones((5, 5), "uint8") 

    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(frameROI, frameROI, mask = red_mask) 

    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
	
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour)
        print(area)
        if(area > 1000): 
            x, y, w, h = cv2.boundingRect(contour) 
            frameROI = cv2.rectangle(frameROI, (x, y),(x + w, y + h),(0, 0, 255), 2) 
			
            cv2.putText(frameROI, "Red", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))
            print('true')
        
        else:
            print("false")

        time.sleep(0.5)


    
    # cv2.imshow('ROI', frameROI)
    # cv2.moveWindow('ROI',640,0)
    # cv2.imshow('webcam', frame)
    # cv2.moveWindow('webcam',0,0)


    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cam.release()