from __future__ import print_function

import cv2
import numpy
import sys
import math
import time

# CONFIGURATION VARIABLES ----------------------------------------------------
MEME_BASE = "trump.mp4"
THRESH = 180
MORPH_W_DIV = 62
MORPH_H_DIV = 130
MAX_VAL = 255
TIME_DELAY = 0.050
CENTROID_WEIGHT = 1
CORNER_WEIGHT = 1
# debug
DEBUGMODE    = True
DEBUGWAIT    = False
DEBUGSAVE    = []
DEBUG_THRESH = False
DEBUG_MORPH  = False
DEBUG_CONT   = False
DEBUG_APPROX = False
DEBUG_CENT   = False
DEBUG_HULL   = False
DEBUG_CORN   = False
DEBUG_HOMOG  = False
DEBUG_MASK   = False # You won't really be able to see it
DEBUG_IMASK  = False # You won't really be able to see this either
DEBUG_MASKED = False
DEBUG_FINAL  = True
# storage
CENTROIDS = []
CORNERS = []
# END CONFIGURATION VARIABLES ------------------------------------------------

def checkCommand():
    if len(sys.argv) != 3:
        print('usage: %s IMAGE1 IMAGE2' % (sys.argv[0]))
        print('usage: %s -v IMAGE2' % (sys.argv[0]))
        sys.exit(1)

def getInputImages():
    return [cv2.imread(sys.argv[1],3),cv2.imread(sys.argv[2],3)]

def getFrames(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    success = True
    while success:
        success,image = cap.read()
        if success:
            frames.append(image)
    cap.release()
    print ("video: %s,    frames: %d" % (filename, len(frames)))
    return frames

def getWH(image):
    return image.shape[1],image.shape[0]

def getMaxContours(contours):
    contour_areas = []
    for i in range(len(contours)):
        current_cont = contours[i]
        current_area = cv2.contourArea(current_cont)
        contour_areas.append(current_area)
    m1 = max(contour_areas)
    index_m1 = contour_areas.index(m1)
    contour_areas[index_m1] = 0
    m2 = max(contour_areas)
    index_m2 = contour_areas.index(m2)
    return contour_areas,[index_m1,index_m2]

def approxPolygon(original_color,contours,pages):
    original_color_with_approx_contour = numpy.copy(original_color)
    approx_contours = []
    for i in range(2):
        epsilon = 0.05*cv2.arcLength(contours[pages[i]],True)
        approx_contours.append(cv2.approxPolyDP(contours[pages[i]],epsilon,True))
        cv2.drawContours(original_color_with_approx_contour, approx_contours, i, (0,255,0), 1)
    return original_color_with_approx_contour,approx_contours

def getConvexHull(original_color,approx_contours):
    original_color_with_convex_hull = numpy.copy(original_color)
    dst = []
    for i in range(2):
        hull = cv2.convexHull(approx_contours[i])
        dst.append(hull)
        cv2.polylines(original_color_with_convex_hull,[hull],True,(0,0,255))
    return dst,original_color_with_convex_hull

def getHomography(original_color,input_images,dst,width,height):
    # Prepare base images for homography
    sum_of_warped = numpy.copy(original_color)
    warped = [numpy.copy(original_color), numpy.copy(original_color)]
    # Homography
    warped_imgs = []
    for i in range(2):
        f_width,f_height = getWH(input_images[i])
        corners = numpy.array(  [ [[0,0]],
                                [[f_width,0]], 
                                [[f_width,f_height]],
                                [[0,f_height]] ], dtype='float32')
        H, mask = cv2.findHomography(corners, dst[i])
        cv2.warpPerspective(input_images[i], H, (width, height), warped[i])
    sum_of_warped = warped[0] + warped[1]
    return sum_of_warped

def getMask(original_color,pages,contours):
    mask = numpy.zeros_like(original_color)
    inv_mask = numpy.zeros_like(original_color)
    inv_mask[:] = 1
    for i in range(2):
        cv2.fillPoly(mask, [contours[pages[i]]], (1,1,1))
        cv2.fillPoly(inv_mask, [contours[pages[i]]], (0,0,0))
    return mask,inv_mask

def showContours(original_color,contours,pages):
    original_color_with_contour = numpy.copy(original_color)
    cv2.drawContours(original_color_with_contour, contours, pages[0], (0,255,0), 1)
    cv2.drawContours(original_color_with_contour, contours, pages[1], (0,255,0), 1)
    DEBUGVIEW(original_color_with_contour)


def getCentroid(contours,pages):
    C = []
    for i in range(len(pages)):
        M = cv2.moments(contours[pages[i]])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        C.append([cx,cy])
    centroids = numpy.array(  [ [ C[0][0],C[0][1] ],
                             [ C[1][0],C[1][1] ] ], dtype='float32')
    return centroids

def drawCentroids(img, C):
    for i in range(1,len(C)):
        cv2.circle(img,(C[i][0,0],C[i][0,1]),CENTROID_WEIGHT/2,(255,0,0),-1)
        cv2.circle(img,(C[i][1,0],C[i][1,1]),CENTROID_WEIGHT/2,(0,0,255),-1)
        cv2.line(img,(C[i-1][0,0],C[i-1][0,1]),(C[i][0,0],C[i][0,1]),(255,0,0),CENTROID_WEIGHT)
        cv2.line(img,(C[i-1][1,0],C[i-1][1,1]),(C[i][1,0],C[i][1,1]),(0,0,255),CENTROID_WEIGHT)

def drawCorners(img, C):
    for i in range(1,len(C)):
        for j in range(4):
            colorone = (255,60*j,30*j)
            colortwo = (30*j,60*j,255)
            cv2.circle(img,(C[i][0][j,0],C[i][0][j,1]),CORNER_WEIGHT,colorone,-1)
            cv2.circle(img,(C[i][1][j,0],C[i][1][j,1]),CORNER_WEIGHT,colortwo,-1)
            cv2.line(img,(C[i-1][0][j,0],C[i-1][0][j,1]),(C[i][0][j,0],C[i][0][j,1]),colorone,CORNER_WEIGHT)
            cv2.line(img,(C[i-1][1][j,0],C[i-1][1][j,1]),(C[i][1][j,0],C[i][1][j,1]),colortwo,CORNER_WEIGHT)

def reorderCentroid(prev_cent, cur_cent):
    cur_cent_t = cur_cent.transpose()
    diff = prev_cent[:,:,None] - cur_cent_t[None,:,:]
    d = numpy.sqrt((diff**2).sum(axis=1))
    if d[0,0] < d[0,1]:
        return False,cur_cent
    else:
        return True,numpy.array(  [ [ cur_cent[1,0],cur_cent[1,1] ],
                                  [   cur_cent[0,0],cur_cent[0,1] ] ], dtype='float32')

def reorderCorners(prev_corners, cur_corners):
    for i in range(len(prev_corners)):
        prev_corners[i] = prev_corners[i].reshape(4,2)
        cur_corners[i] = cur_corners[i].reshape(4,2)
        cur_corners_t = cur_corners[i].transpose()
        diff = prev_corners[i][:,:,None] - cur_corners_t[None,:,:]
        d = (diff**2).sum(axis=1)
        minimum = numpy.argmin(d, axis=0)
        smallest = d[0][0] + d[1][1] + d[2][2] + d[3][3]
        smallest_idx = 0
        for j in range(4):
            sqsum = d[0][j] + d[1][(j+1)%4] + d[2][(j+2)%4] + d[3][(j+3)%4]
            if sqsum < smallest:
                smallest = sqsum
                smallest_idx = j
        cur_corners[i] = numpy.array( [ cur_corners[i][smallest_idx],
                                        cur_corners[i][(smallest_idx+1)%4],
                                        cur_corners[i][(smallest_idx+2)%4],
                                        cur_corners[i][(smallest_idx+3)%4] ], dtype='float32')
    return cur_corners

def DEBUGVIEW(image):
    if DEBUGMODE:
        cv2.imshow("window", image)
        if DEBUGWAIT:
            cv2.waitKey(0)
        else:
            DEBUGSAVE.append(image)

def main():
    checkCommand()
    if sys.argv[1] == "-v":
        input_video = getFrames(sys.argv[2])
        IMAGE = False
    else:
        input_images = getInputImages()
        IMAGE = True

    # Get frames from video
    frames = getFrames(MEME_BASE)

    # Height, Width
    width,height = getWH(frames[0])

    # Modify each frame
    for idx in range(len(frames)):
        original_color = frames[idx]
        original_bw = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)

        # Thresholding
        th,thresholded = cv2.threshold(original_bw, THRESH, MAX_VAL, cv2.THRESH_BINARY)
        if DEBUG_THRESH: DEBUGVIEW(thresholded)

        # Morphological operators
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width/MORPH_W_DIV,height/MORPH_H_DIV))
        morphed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        if DEBUG_MORPH: DEBUGVIEW(morphed)

        # Find contours calculate contour areas and draw page contours
        cont, contours, hierarchy = cv2.findContours(morphed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_areas,pages = getMaxContours(contours)
        if DEBUG_CONT: showContours(original_color,contours,pages)

        # Contour Approximation to remove hand
        original_color_with_approx_contour,approx_contours = approxPolygon(original_color,contours,pages)
        if DEBUG_APPROX: DEBUGVIEW(original_color_with_approx_contour)

        # Find centroid
        cur_cent = getCentroid(contours,pages)
        # Rearrange centroid array is needed
        if idx > 0:
            switch,cur_cent = reorderCentroid(CENTROIDS[idx-1], cur_cent)
        CENTROIDS.append(cur_cent)

        drawCentroids(original_color_with_approx_contour, CENTROIDS)
        if DEBUG_CENT: DEBUGVIEW(original_color_with_approx_contour)

        # Find bounding points using convex hull
        dst,original_color_with_convex_hull = getConvexHull(original_color,approx_contours)
        if DEBUG_HULL: DEBUGVIEW(original_color_with_convex_hull)

        # Swap if centroid tracking indicates points need to be switched
        if idx > 0 and switch:
            dst = dst[::-1]

        # Rearrange corner points if needed
        if idx > 0:
            dst = reorderCorners(CORNERS[idx-1], dst)
        CORNERS.append(dst)
        drawCorners(original_color_with_convex_hull, CORNERS)
        if DEBUG_CORN: DEBUGVIEW(original_color_with_convex_hull)

        # Image or video mode
        if IMAGE:
            sum_of_warped = getHomography(original_color,input_images,dst,width,height)
        else:
            # v_width = input_video[idx].shape[1]
            # v_height = input_video[idx].shape[0]
            # input_images = [input_video[idx][0:v_height, 0:(v_width)/2], input_video[idx][0:v_height,(v_width/2):v_width]]
            input_images = [input_video[idx], input_video[idx]]
            sum_of_warped = getHomography(original_color,input_images,dst,width,height)
        if DEBUG_HOMOG: DEBUGVIEW(sum_of_warped)

        # Create mask using contour
        mask,inv_mask = getMask(original_color,pages,contours)
        if DEBUG_MASK: DEBUGVIEW(mask)
        if DEBUG_IMASK: DEBUGVIEW(inv_mask)

        # Masking
        sum_of_warped *= mask
        if DEBUG_MASKED: DEBUGVIEW(sum_of_warped)

        # Add with original color
        original_color *= inv_mask
        original_color += sum_of_warped
        if DEBUG_FINAL: DEBUGVIEW(original_color)

    if len(DEBUGSAVE)>0:
        while True:
            for img in DEBUGSAVE:
                cv2.imshow('window', img)
                cv2.waitKey(15)
                time.sleep(TIME_DELAY)
    cv2.destroyAllWindows()
    return

main()
