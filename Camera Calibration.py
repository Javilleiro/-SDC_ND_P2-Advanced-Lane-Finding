# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:06:42 2021

@author: dahouse
"""

import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


#Function to perform a Camera Calibration
    #Inputs a list of cañ_images, nx and ny to find in a chessboard.
def  cal_cam(images,nx,ny):
    #Arrays to store object points and image points from all the images
    objpoints = [] #3D points in real world space
    imgpoints = [] # 2D points in image plane
      
    #Prepare object points (3 coordinates in each x,y,z)
    objp = np.zeros((ny*nx,3), np.float32) #matrix of zeros
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #☺x, y coordinates
    
    for fname in images:
        #read in each image
        img = mpimg.imread(fname)
        
        #convert img to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny),None)
        
        # If found, add object points and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
        
            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            
    #Calibrate the Camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return ret, mtx, dist


###################CAMERA CALIBRATION######################################
'''
This Section includes:
    The variables needed to perform a camera calibration.
    The Outputs of the Calibration performed with the provided Cal_Images
    2 Images (Original vs. Undistorted). Part of Project Rubric. 
'''

# Write the number of inside corners of the Chessboard used to Calibrate.
nx = 9 #Inside corners in x
ny = 6 #Inside corners in y
# Read in and make a list of calibration images
cal_imgs = glob.glob('camera_cal/calibration*.jpg')

ret, mtx, dist = cal_cam(cal_imgs,nx,ny)

'''
#The following parameters were obteined by calibrationg the camera with the cal_images
ret = 0.9230912642921472
mtx = np.array([[1.15660712e+03, 0.00000000e+00, 6.68960302e+02],
      [0.00000000e+00, 1.15164235e+03, 3.88057002e+02],
      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.23185386, -0.11832054, -0.00116561,  0.00023902,  0.15356159]])
'''

#print a cal_image (original vs undistorted) to test the dist & mtx
#org = mpimg.imread('camera_cal/calibration1.jpg')
org = mpimg.imread('test_images/test3.jpg')
dst = cv2.undistort(org, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(org, cmap ='gray')
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dst, cmap ='gray')
ax2.set_title('Final Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
#############################################################################