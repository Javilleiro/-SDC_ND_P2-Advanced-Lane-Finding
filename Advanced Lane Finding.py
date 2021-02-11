# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:13:37 2021

@author: dahouse
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import line

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML



#Function to print 2 images (original & Final).
def print_img(o_image, f_image):  
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(o_image, cmap ='gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(f_image, cmap ='gray')
    ax2.set_title('Final Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.) 
                
    return


# Function to create a thresholded binary image
def binary_image(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
       
    combined = np.zeros_like(sxbinary)
    combined[(sxbinary == 1) | (s_binary==1)] = 1
    #plt.imshow(color_binary)
    return combined

def warp_image(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[590,450],[690,450],[180,img_size[1]],[1110,img_size[1]]])
    dst = np.float32([[250,0],[img_size[0]-250,0],[250,img_size[1]],[img_size[0]-250,img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    #print_img(img, warped)    
    return warped, M, Minv

def histogram(img):
    
    #grab only the bottom half of the image
    bottom_half = img[img.shape[0]//2:,:]
    
    #Sum across image pixel vertically
    histogram = np.sum(bottom_half, axis=0)
    #plt.plot(histogram)
    
    return histogram

def get_peaks(hist):
    
    midpoint = np.int(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    
    return leftx_base, rightx_base

def find_lane_pixels(binary_warped, leftx_base, rightx_base, nwindows=10, margin=100, minpix=50):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    window_height = np.int(binary_warped.shape[0]//nwindows) #height of the windows
    
    #identify the x and y positions of all the activated pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Current Positions (will be updated for each window)
    lfx_current = leftx_base 
    rgx_current = rightx_base 
    
    #Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        #Identify window boundaries in x and y
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = lfx_current - margin
        win_xleft_high = lfx_current + margin
        win_xright_low = rgx_current - margin
        win_xright_high = rgx_current + margin
        
        #Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0),2)
        
        #Identify the activated ìxels in x and y inside the window
        lf_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
        rg_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]
        
        #Appends the indices to the lists
        left_lane_inds.append(lf_inds)
        right_lane_inds.append(rg_inds)
        
        if len(lf_inds) > minpix:
            lfx_current = np.int(np.mean(nonzerox[lf_inds]))
            
        if len(rg_inds) > minpix:
            rgx_current = np.int(np.mean(nonzerox[rg_inds]))
            
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #Extract left and right line pizels positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]        
        
    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped, lfx_pos, lfy_pos, rgx_pos, rgy_pos):
    #Find the second order polynomial to each line
    left_fit = np.polyfit(lfy_pos,lfx_pos,2)
    right_fit = np.polyfit(rgy_pos,rgx_pos,2)
    
    # Generate y values from top to bottom
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    #Generate x values with the 2nd order polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    '''
    ## Visualizaion Steps ##
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #set Colors in the left and right lane regions
    out_img[lfy_pos,lfx_pos] = [255,0,0]
    out_img[rgy_pos,rgx_pos] = [0,0,255]
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(out_img)
    ## End visualization steps ##
    '''
    return left_fit, right_fit, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Set the area of search based on activated x-values
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, img_fit = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
    
    # Generate y values from top to bottom
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    #Generate x values with the 2nd order polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    '''
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(binary_warped)
    # Color in left and right line pixels
    binary_warped[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    binary_warped[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(binary_warped, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
   '''
    return leftx, lefty, rightx, righty, left_fit, right_fit, ploty

def measure_curvature_pixels(left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**(3/2))/(np.absolute(2*left_fit[0]))  
    right_curverad = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**(3/2))/(np.absolute(2*right_fit[0]))  
                                                                           
    return left_curverad, right_curverad

def measure_curvature_real(lfx_pos, lfy_pos, rgx_pos, rgy_pos, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    
    left_fit_cr = np.polyfit(lfy_pos*ym_per_pix, lfx_pos*xm_per_pix, 2)
    right_fit_cr = np.polyfit(rgy_pos*ym_per_pix, rgx_pos*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### radius of curvature 
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2))/(np.absolute(2*left_fit_cr[0]))
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2))/(np.absolute(2*right_fit_cr[0]))
    
    return left_curverad, right_curverad

def car_offset(binary_warped, left_fit, right_fit, ploty):
    
    y = len(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #Get the X Value (Center) from the polynomials of both lines
    leftx_base = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    rightx_base = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    
    midlane = (leftx_base + rightx_base)/2
    
    midimg = binary_warped.shape[1]/2
    
    offset_pix = midimg - midlane 
    offset_mts = offset_pix*xm_per_pix
    
    return offset_mts

def sanity_checks(lfx_pos, lfy_pos, rgx_pos, rgy_pos, left_fit, right_fit, ploty, xm_per_pix, ym_per_pix):
    #Compute fits in both lines with calculated left and right fits
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    #Sanity check of lane width
    #Compute the lane width
    lane_width = np.absolute((left_fitx - right_fitx)*xm_per_pix)  
    avg_lane = np.mean(lane_width)
    
    #Compute the curvature of the road in meters
    lf_curv_mts, rg_curv_mts = measure_curvature_real(lfx_pos, lfy_pos,
                               rgx_pos, rgy_pos, ploty)
    
    #Calculate direction of the curve   
    sll = left_fitx[719] - left_fitx[0] 
    slr = right_fitx[719] - right_fitx[0]
    
    if avg_lane > 3.5 and avg_lane < 4.0 and ((sll > 0 and slr > 0) or (sll < 0 and slr < 0)):
        detection = True
    else:
        detection = False

    
    return detection, lf_curv_mts, rg_curv_mts, avg_lane

def process_image(image):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #undistort image with the parameters obtained with the calibration
    dst = cv2.undistort(image, mtx, dist, None, mtx) #distorted image
    #Get combined binary image with color and Gradient thresholds
    binary_img = binary_image(dst, (170,230),(20,100))
    #Warp image to get an eye-birds image
    warped, M, Minv = warp_image(binary_img)
    
       
    if left_line.detected == True and right_line.detected == True:
        ## Search from Prior ## 
        #When we get the first curves with an image of the video, the next frame
        #we can search just between a margin without implementing sliding windows
        lfx_pos, lfy_pos, rgx_pos, rgy_pos, left_fit, right_fit, ploty = search_around_poly(warped, 
                                                                         left_line.current_fit, right_line.current_fit)
        
        good_detection, lf_curv_mts, rg_curv_mts, avg_lane = sanity_checks(lfx_pos, lfy_pos, rgx_pos, rgy_pos, 
                                                     left_fit, right_fit, ploty, xm_per_pix, ym_per_pix)
        left_line.detected = good_detection
        right_line.detected = good_detection
        
    if left_line.detected == False  or right_line.detected == False:
        ## Histogram Peaks and Sliding Windows Method ##
        #Get the histogram of the bottom half of the image.
        hist = histogram(warped)
        #Get the peaks of the histograms and get the starting points of the lane lines.
        lfx_base, rgx_base = get_peaks(hist)
        #Set HYPERPARAMETERS
        nwindows = 9 #number of sliding windows in the image
        margin = 100 #Width of the windows +/- margin
        minpix = 50 #Min number of pixels to recenter window
        #Call a function that performs the detection of the pixels of both lane lines
        lfx_pos, lfy_pos, rgx_pos, rgy_pos = find_lane_pixels(warped, lfx_base, rgx_base, nwindows, margin, minpix)
        #Compute a 2nd order polynomial to get the curvature of the lane
        left_fit, right_fit, ploty = fit_polynomial(warped, lfx_pos, lfy_pos, rgx_pos, rgy_pos)
        ## End of the Sliding windows method ##

        good_detection, lf_curv_mts, rg_curv_mts, avg_lane = sanity_checks(lfx_pos, lfy_pos, rgx_pos, rgy_pos,
                                                     left_fit, right_fit, ploty, xm_per_pix, ym_per_pix)
        
        left_line.detected = good_detection
        right_line.detected = good_detection
    
    #Compute fits in both lines with calculated left and right fits
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if left_line.detected == True or left_line.radius_of_curvature == None:
        # Saving X and Y values
        left_line.allx = lfx_pos
        left_line.ally = lfy_pos
        right_line.allx = rgx_pos
        right_line.ally = rgy_pos 
        #Saving Left and Right Fit
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
        #Save Curvatures
        left_line.radius_of_curvature = lf_curv_mts
        right_line.radius_of_curvature = rg_curv_mts
        #Save Fit
        left_line.recent_xfitted = left_fitx
        right_line.recent_xfitted = right_fitx
        
    else:
        #Saving Left and Right Fit
        left_line.current_fit = (left_line.current_fit + left_fit)/2
        right_line.current_fit = (right_line.current_fit + right_fit)/2
        #Save Curvatures
        left_line.radius_of_curvature = (left_line.radius_of_curvature+lf_curv_mts)/2
        right_line.radius_of_curvature = (right_line.radius_of_curvature+rg_curv_mts)/2
        #Save Fit
        left_line.recent_xfitted = (left_line.recent_xfitted+left_fitx)/2
        right_line.recent_xfitted = (right_line.recent_xfitted+right_fitx)/2

    #compute the average of the both lines to obtain the curvature of the lane
    curv_avg = (left_line.radius_of_curvature + right_line.radius_of_curvature)/2     
    
    #Compute the offset of the car from the center of the lane
    offset = car_offset(warped, left_line.current_fit, right_line.current_fit, ploty)
       
    #Evaluate if the car has an offset to the right or to the left
    if offset <= 0:
        side = 'left'
    else:
        side = 'right'
        
    offset = '%.2f' % np.absolute(offset)
    
    ## DRAWING ##
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.recent_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.recent_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    #Write the Radius of Curvature in the top of the image
    cv2.putText(result, 'Radius of Curvature = ' + str(int(curv_avg)) + 'm', (35,75), 
               cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,0), 2)
    #Write the Offset of the Car from the center of the lane
    cv2.putText(result, 'Vehicle is ' + str(offset) + 'm ' + side + ' of center', (35,150),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    

    ## END OF DRAWING ##   

    return result    


#Camera Matrix and Disrtortion Coefficients obtained with Camera Calibration.py
ret = 0.9230912642921472
mtx = np.array([[1.15660712e+03, 0.00000000e+00, 6.68960302e+02],
      [0.00000000e+00, 1.15164235e+03, 3.88057002e+02],
      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.23185386, -0.11832054, -0.00116561,  0.00023902,  0.15356159]])

left_line = line.Line()
right_line = line.Line()


'''
test_images = glob.glob('test_images/test*.jpg')
for image in test_images:
    img = mpimg.imread(image)
    test = binary_image(img, (150,230),(30,100))
    warped, M, Minv = warp_image(test)
    hist = histogram(warped)
    left_base, right_base = get_peaks(hist)
    leftx, lefty, rightx, righty = find_lane_pixels(warped, left_base, right_base)
    left_fit, right_fit, ploty = fit_polynomial(warped, leftx, lefty, rightx, righty)
    ## Visualizaion Steps ##
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped, warped, warped))*255
    #set Colors in the left and right lane regions
    out_img[lefty,leftx] = [255,0,0]
    out_img[righty,rightx] = [0,0,255]
    
    #Generate x values with the 2nd order polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(out_img)
    plt.show()
    ## End visualization steps ##
'''

'''
image = mpimg.imread('test_images/test8.jpg')
result = process_image(image)
plt.imshow(result)

'''

#Function taken from the First Project
white_output = 'project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("project_video.mp4").subclip(36,48)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)


