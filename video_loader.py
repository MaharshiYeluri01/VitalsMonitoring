import cv2
import numpy as np
import os,shutil
import glob
from configs import *
import os.path as op
from datetime import datetime

def readVideo(fp,crop_cords=None,extended_cords=None):
    cap = cv2.VideoCapture(fp)
    cropped_frames=[];frames=[]
    if crop_cords:
        x1,y1,x2,y2=crop_cords
    if extended_cords:
        ex1,ey1,ex2,ey2=extended_cords
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if crop_cords:
                image=frame[y1:y1+y2,x1:x1+x2]
                cropped_frames.append(image)
            if extended_cords:
                padded_image=frame[ey1:ey1+ey2,ex1:ex1+ex2]
                frames.append(padded_image)
            else:
                frames.append(frame)
        else: 
            break
    
    cap.release()
    if extended_cords:
        return np.array(frames),np.array(cropped_frames)
    return np.array(frames)

def boxLocation(w,h,x_pad=BOX_HPAD,y_pad=BOX_VPAD):
    BW=220*w//600
    BH=300*h//400
    X=w//3
    Y=h//6

    pw=int(x_pad*BW)
    ph=int(y_pad*BH)

    box=(X,Y,BW,BH)
    extended_box=(X-pw,Y-ph,BW+2*pw,BH+2*ph)
    return box,extended_box
