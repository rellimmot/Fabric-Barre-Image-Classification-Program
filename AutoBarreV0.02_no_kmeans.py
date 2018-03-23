# -*- coding: utf-8 -*-
"""

07/28/2017:	v0.01 trying to make it automatic

"""
from Tkinter import *
import ttk
import cv2
import os
import numpy as np
import math
import time
import csv

#finds worst case size we will crop to after rotation; 0.35 to approximate sqrt(2)/4
def findCropSize( raw ):
    rows, cols = raw.shape[0:2]
    print rows, cols
    cropsize = 0.35*cols
    if rows < cols:
        cropsize = 0.35*rows
    return cropsize

#rotates raw image by theta degrees, returns rotated image
def rotateTheta( raw, theta ):
    imgcenter = tuple(np.array(raw.shape[0:2])/2)
    M = cv2.getRotationMatrix2D((imgcenter[1],imgcenter[0]), theta, 1.0)
    rotated = cv2.warpAffine(raw, M, (raw.shape[1],raw.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated
	
#crops image to centered square of specified size, returns cropped image
def autoCrop ( raw , cropsize ):
    rows, cols = raw.shape[0:2]
    cropped = raw[math.ceil(rows*0.5 - cropsize):math.floor(rows*0.5 + cropsize),math.ceil(cols*0.5 - cropsize):math.floor(cols*0.5 + cropsize)]
    return cropped

#detect - polyfit to remove brightness curve from flash, equalize, kmeans, etc
def detect( raw ):
    #raw = equalize(raw) #moved to own function
    rows, cols = raw.shape
    mean = np.mean(raw, axis = 1)      #mean is a 1D type (row vector)  #creates vector of mean of each row of pixels?
    mean = np.float32(mean)
    #print raw.shape
    x = np.array(range(len(mean)))
    y = mean
    z = np.polyfit(x,y,2)    #returns vector of coefficiencts for degree 2 polynomial
    mean = [v-(z[0]*ind**2+z[1]*ind+z[2]) for ind,v in enumerate(y)]   #enumerate puts 0,1,2.. into ind and means into v
    mean = np.float32(mean)  #mean now contains error from polyfit to mean?
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.1) #max iterations and/or desired accuracy
    flags = cv2.KMEANS_RANDOM_CENTERS #let opencv choose random centers for clusters
    initial_ram = 5 #actually this is number of attempts
    #the depth of data must be 32F, that is np.float32
    error,labels,centers = cv2.kmeans(mean,2,None,criteria,initial_ram,flags) 
    #print error #aka "compactness"
    if centers[1] < centers[0]:
        labels = [ 1-x for x in labels]
    labels = [ x*255 for x in labels]
    #print labels
    #lables are column vector
    RowMean = np.tile(labels,[1,cols])   #the tile function is what makes the bars go all the way across by repeating "labels" numcol times
    #RowMean = RowMean.T
    detected = RowMean.astype(np.uint8)
    
    imgOut = np.zeros((raw.shape[0],raw.shape[0]*2),np.uint8)
    imgOut[:raw.shape[0], :raw.shape[0]] = detected
    imgOut[:raw.shape[0], raw.shape[0]:raw.shape[0]*2] = raw
    
    return (imgOut, error)

#predetect - exerything from detect except kmeans, use to make a means array for output to analyze elsewhere
def preDetect( raw ):
    #raw = equalize(raw) #moved to own function
    rows, cols = raw.shape
    mean = np.mean(raw, axis = 1)      #mean is a 1D type (row vector)  #creates vector of mean of each row of pixels?
    mean = np.float32(mean)
    #print raw.shape
    x = np.array(range(len(mean)))
    y = mean
    z = np.polyfit(x,y,2)    #returns vector of coefficiencts for degree 2 polynomial
    mean = [v-(z[0]*ind**2+z[1]*ind+z[2]) for ind,v in enumerate(y)]   #enumerate puts 0,1,2.. into ind and means into v
    mean = np.float32(mean)  #mean now contains error from polyfit to mean?
    
    #criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.1) #max iterations and/or desired accuracy
    #flags = cv2.KMEANS_RANDOM_CENTERS #let opencv choose random centers for clusters
    #initial_ram = 5 #actually this is number of attempts
    #the depth of data must be 32F, that is np.float32
    #error,labels,centers = cv2.kmeans(mean,2,None,criteria,initial_ram,flags) 
    #print error #aka "compactness"
    #if centers[1] < centers[0]:
    #    labels = [ 1-x for x in labels]
    #labels = [ x*255 for x in labels]
    #print labels
    #lables are column vector
    #RowMean = np.tile(labels,[1,cols])   #the tile function is what makes the bars go all the way across by repeating "labels" numcol times
    #RowMean = RowMean.T
    #detected = RowMean.astype(np.uint8)
    
    #imgOut = np.zeros((raw.shape[0],raw.shape[0]*2),np.uint8)
    #imgOut[:raw.shape[0], :raw.shape[0]] = detected
    #imgOut[:raw.shape[0], raw.shape[0]:raw.shape[0]*2] = raw
    
    return mean
    
def equalize(img):
    if np.ndim(img) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img

def autorotate( raw, cropsize ):
    maxError = 0;
    thetaAtMaxError = 0;
    for theta in xrange(0,180):
        tempImg = rotateTheta ( raw, theta )
        tempImg = autoCrop (tempImg,cropsize)
        (detected, error) = detect(tempImg)
        #cv2.imshow('%s' %img_name,detected)
        #cv2.waitKey(0)
        if maxError < error:
            maxError = error
            thetaAtMaxError = theta
    return (maxError, theta)    
    
####################start here#############################
#change the directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('data')

#open the image
img_names = ['AATCC_Uniformity_1C.JPG',
             'AATCC_Uniformity_2C.JPG',
             'AATCC_Uniformity_3C.JPG',
             'AATCC_Uniformity_4C.JPG',
             'AATCC_Uniformity_5C.JPG',
             'AATCC_Uniformity_6C.JPG',
             'AATCC_Uniformity_7C.JPG',
             'AATCC_Uniformity_8C.JPG',
             'AATCC_Uniformity_9C.JPG']





with open('outputmeans.csv', 'wb') as csvfile:
    #create csv-writer-object
    outputwriter = csv.writer(csvfile, delimiter=',',
                              quotechar="'", quoting=csv.QUOTE_MINIMAL)
                              
    #write header row
    outputwriter.writerow(["number of blurs","compactness (image 1 through 9)"])
    
    temperror = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    errors = [temperror,temperror,temperror,temperror,temperror,temperror,temperror,temperror,temperror]

    for j in xrange(0,9):
        i = 0
        raw = cv2.imread(img_names[j])

        #equalize
        raw = equalize(raw)

        #find crop size
        cropsize = findCropSize( raw )
        cropsize = 500

        #find rotation angle
        (maxError, thetaAtMaxError) = autorotate( raw, cropsize )

        #rotate the image
        #tempImg = rotateTheta ( raw, thetaAtMaxError + 90 ) #for real images
        tempImg = rotateTheta ( raw, thetaAtMaxError) #for standards images
        tempImg = autoCrop (tempImg,cropsize)
               
        #detect barre
        #(detected, temperror[i]) = detect(tempImg)
        #write means to csv
        outputwriter.writerow(preDetect(tempImg))
        
        #cv2.imshow('%s' %img_names[j],detected)
        #cv2.waitKey(0)
        blurTemp = tempImg
        
        #print "compactness after", 0, "blurs is", temperror[i]
        
        #iterate a gaussian blur, writing each new kmeans compactness as new row
        #for i in xrange(1,21):
            #blur the image
            #blurTemp = cv2.GaussianBlur(blurTemp,(5,5),0)
            
            #detect barre
            #(detected, temperror[i]) = detect(blurTemp)
            #print "compactness after", i, "blurs is", temperror[i]
            #outputwriter.writerow(preDetect(blurTemp))
            #cv2.imshow('%s' %img_names[i],detected)
            #cv2.waitKey(0)
        #errors[j] = temperror
        #cv2.imwrite('1C_blur-9.JPG',detected)
        #display and write to csv
        #outputwriter.writerow([i,errors[j]])
    #for i in xrange(0,21):
        #outputwriter.writerow([i,errors[0:8][i]])