
import os
from tqdm import tqdm
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from signal_processing import *
from params import *

bgr2rgb=lambda image:cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def get_video_files(folder,extensions=EXTENSIONS):
    return [folder+'/'+f for f in os.listdir(folder) if os.path.splitext(f)[-1] in extensions]

def filter_by_size(path,min_size_in_mb=3):
    filesize=size_stat(path)
    if filesize>=min_size_in_mb:return True
    return False

def videoList(path):
    video_files=get_video_files(path)

    return [f for f in video_files if filter_by_size(f) ]

size_stat=lambda path:Path(path).stat().st_size/pow(10,6)

def bgrBPM(signal,seconds=SECONDS):
    fps=signal.shape[0]/seconds

    gc_result=g_channel(signal,fps)
    return gc_result


def estimateBPM(signal,seconds=SECONDS):
    fps=signal.shape[0]/seconds
    ica_result=ICA(signal,fps)
    gc_result=g_channel(signal[:,1],fps)
    gc_result.update(ica_result)
    return gc_result

def mov_avg(x, window=5):
    if window==1:return x
    avg=[sum(np.ones(window) * x[m:m+window]) / window for m in range(len(x)-(window-1))]
    n=window//2

    if window%2==0:
        return [avg[0]]*(n-1)+avg+[avg[-1]]*n
    else:
        return [avg[0]]*n+avg+[avg[-1]]*n

def sigmoid_softmax(x):
    x=np.array(x)
    return SM(1/(1 + np.exp(-x)))


def plotFFT(res,img=None,file_name='temp',gt=None,window=1):
    columns=4
    if img:
        columns=5
        f, ax = plt.subplots(1,5,figsize=(24,4))
        (ax0,ax1, ax2,ax3,ax4)=ax.flatten()
        ax0.imshow(bgr2rgb(img))
    else:
        f, ax = plt.subplots(1,4,figsize=(24,4))
        (ax1, ax2,ax3,ax4)=ax.flatten()
    
    ax1.plot(res['gf'],mov_avg(res['gw'],window),color='green')
    ax2.plot(res['Confidence'][0],mov_avg(res['Confidence'][3],window),color='indigo')    
    ax3.plot(res['Confidence'][1],mov_avg(res['Confidence'][4],window),color='red')   
    ax4.plot(res['Confidence'][2],mov_avg(res['Confidence'][5],window),color='blue')
    
    if gt:
        ax1.scatter(gt,[max(res['gw'])+0.2]*len(gt),color='orange')
        ax2.scatter(gt,[max(res['Confidence'][3])+0.3]*len(gt),color='orange')
        ax3.scatter(gt,[max(res['Confidence'][4])+0.3]*len(gt),color='orange')
        ax4.scatter(gt,[max(res['Confidence'][5])+0.3]*len(gt),color='orange')
    plt.xlabel('BPM')
    plt.ylabel('Weight')
    plt.title(file_name+f'moving avg {window}')
    