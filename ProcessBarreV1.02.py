# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 15:46:41 2016

@author: kang, shurlds, miller

V1.01, 2016-09-06: File extension no longer case sensitive. --R. Walker Shurlds, 
V1.02, 2017-07-13: Fixed spelling of "equalize". --R. Walker Shurlds, 
"""
from Tkinter import *
import ttk
import cv2
import os
import numpy as np
import math

os.chdir(os.path.dirname(os.path.realpath(__file__)))

os.chdir('data')
 
class window(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.grid()                  #from Tkinter, "organizes widgets in a table-like structure in the parent widget"
        #self.pack()
        self.createCombobox()
        self.createButton()
        
        
        
    def createCombobox(self):
        ext = [".jpg", ".JPG"]
        self.v_list=[ f for f in os.listdir(os.getcwd()) if f.endswith(tuple(ext))]
        self.combo=ttk.Combobox(self, values=self.v_list)
        self.combo.grid(row=3,columnspan=3, sticky=W+E+N+S)
        #self.combo.pack(side='bottom',fill='x')
        self.combo.bind('<<ComboboxSelected>>', self.displayimage)
    
    def displayimage(self, event):
        img_name=self.combo.get()
        img = cv2.imread(img_name)
        cv2.imshow('%s' %img_name,img)
        
    def createButton(self):
        self.down_button=ttk.Button(self, text='downsample')
        #self.down_button.pack(side='left')
        self.down_button.grid(row=0, column=0, sticky=W+E+N+S)
        self.down_button['command']=self.downsample
        
        self.up_button=ttk.Button(self, text='upsample')
        #self.up_button.pack(side='left')
        self.up_button.grid(row=0, column=1, sticky=W+E+N+S)
        self.up_button['command']=self.upsample
        
        self.quit_button=ttk.Button(self, text='destroy all windows')
        #self.quit_button.pack(side='bottom')
        self.quit_button.grid(row=2, column=0, columnspan=2, sticky=W+E+N+S)
        self.quit_button['command']=self.destroyAllWindows
        
        self.crop_button=ttk.Button(self, text='crop image')
        #self.crop_button.pack(side='right')
        self.crop_button.grid(row=1, column=0, sticky=W+E+N+S)
        self.crop_button['command']=self.crop
        
        self.rotate_button=ttk.Button(self, text='rotate image')
        #self.rotate_button.pack(side='right')
        self.rotate_button.grid(row=1, column=1, sticky=W+E+N+S)
        self.rotate_button['command']=self.rotate       
        
        self.update=ttk.Button(self,text='update image')
        #self.update.pack(side='right')
        self.update.grid(row=0, column=2, sticky=W+E+N+S)
        self.update['command']=self.updateList
        
        
        self.detector=ttk.Button(self,text='detect stripe')
        #self.detector.pack(side='left')
        self.detector.grid(row=1, column=2, sticky=W+E+N+S)
        self.detector['command']=self.detect
        
        self.equalizer=ttk.Button(self,text='equalize image')
        #self.equalizer.pack(side='right')
        self.equalizer.grid(row=2, column=2, sticky=W+E+N+S)
        self.equalizer['command']=self.equalize
        

    def downsample(self):
        img_name=self.combo.get()
        
        img = cv2.resize(cv2.imread(img_name), (0,0), fx=0.5, fy=0.5)
        img_name = img_name[:-4]+'_down'+'.jpg'
        self.v_list.append(img_name)
        self.combo['values']=self.v_list
        cv2.imshow('%s' %img_name,img)
        cv2.imwrite(img_name,img)
        
    def upsample(self):
        img_name=self.combo.get()
        img = cv2.resize(cv2.imread(img_name), (0,0), fx=2, fy=2)
        img_name = img_name[:-4]+'_up'+'.jpg'
        self.v_list.append(img_name)
        self.combo['values']=self.v_list
        cv2.imshow('%s' %img_name,img)
        cv2.imwrite(img_name,img)
        
    def destroyAllWindows(self):
        cv2.destroyAllWindows()
        
    def updateList(self):
        self.v_list=[ f for f in os.listdir(os.getcwd()) if f.endswith('jpg')]
        self.combo['values']=self.v_list
        print self.v_list
    
    
    def crop(self):
        img_name=self.combo.get()
        img = cv2.imread(img_name)
        cv2.imshow('%s' %img_name,img)
        cropdata=[]
        cv2.setMouseCallback('%s' %img_name, 
                             lambda event,x,y,flags,param: self.crop_callback(event,x,y,flags,param,\
                             name=img_name,data=cropdata, image=img ))
    

    def crop_callback(self, event,x,y,flags,param,**kw):
        
        img_name=kw['name']
        cropdata=kw['data']
        img=kw['image']
        
        if len(cropdata)>=4:
            pass
        elif event == cv2.EVENT_LBUTTONDOWN:
            cropdata.append(x)
            cropdata.append(y)
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.imshow('%s' %img_name,img)
        elif event == cv2.EVENT_LBUTTONUP:
            cropdata.append(x)
            cropdata.append(y)
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.imshow('%s' %img_name,img)
            raw = cv2.imread(img_name)
            
            img = raw[cropdata[1]:cropdata[3],cropdata[0]:cropdata[2]]
            
            img_name = img_name[:-4]+'_crop'+'.jpg'

            cv2.imshow('%s' %img_name,img)
            cv2.imwrite(img_name,img)
            
        else:
            pass
        
    
    def rotate(self):
        img_name=self.combo.get()
        rotatedata=[]
        img = cv2.imread(img_name)
        cv2.imshow('%s' %img_name,img)
        cv2.setMouseCallback('%s' %img_name, 
                             lambda event,x,y,flags,param: self.rotate_callback(event,x,y,flags,param,\
                             name=img_name,data=rotatedata, image=img) ) 
    

    def rotate_callback(self, event,x,y,flags,param, **kw):
        img_name=kw['name']
        img=kw['image']
        rotatedata=kw['data']
        if len(rotatedata)>=4:
            pass
        elif event == cv2.EVENT_LBUTTONDOWN:
            rotatedata.append(x)
            rotatedata.append(y)
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.imshow('%s' %img_name,img)
            
        elif event == cv2.EVENT_LBUTTONUP:
            rotatedata.append(x)
            rotatedata.append(y)
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.imshow('%s' %img_name,img)
            raw = cv2.imread(img_name)
            
            theta = math.atan( (rotatedata[3]-rotatedata[1])*1.0 / (rotatedata[2]-rotatedata[0]) )
            imgcenter = tuple(np.array(raw.shape[0:2])/2)
            print rotatedata
            print imgcenter
            print theta
            M = cv2.getRotationMatrix2D(imgcenter, theta/3.14*180, 1.0)
            img = cv2.warpAffine(raw, M, (raw.shape[1],raw.shape[0]), flags=cv2.INTER_LINEAR)
            
            img_name = img_name[:-4]+'_rotate'+'.jpg'

            cv2.imshow('%s' %img_name,img)
            cv2.imwrite(img_name,img)
            
        else:
            pass
        
    def detect(self):
        img_name=self.combo.get()
        raw=cv2.imread(img_name)
        if np.ndim(raw) == 3:
            raw = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        rows, cols = raw.shape
        mean = np.mean(raw, axis = 1)      #mean is a 1D type (row vector)  #creates vector of mean of each row of pixels?
        mean = np.float32(mean)
        print raw.shape
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
        if centers[1] < centers[0]:
            labels = [ 1-x for x in labels]
        labels = [ x*255 for x in labels]
        #print labels
        #lables are column vector
        RowMean = np.tile(labels,[1,cols])   #the tile function is what makes the bars go all the way across by repeating "labels" numcol times
        #RowMean = RowMean.T
        img = RowMean.astype(np.uint8)
        
        print labels
        img_name = img_name[:-4]+'_detected'+'.jpg'
        cv2.imshow('%s' %img_name,img)
        cv2.imwrite(img_name,img)
        
    def equalize(self):
        img_name=self.combo.get()
        raw=cv2.imread(img_name)
        if np.ndim(raw)==3:
            raw = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(raw)
        
        
        img_name = img_name[:-4]+'_equalized'+'.jpg'
        cv2.imshow('%s' %img_name,img)
        cv2.imwrite(img_name,img)
        



app=window()

app.mainloop()


