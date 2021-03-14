import cv2
import numpy as np
import os
import shutil
import os.path as op
from sklearn.decomposition import FastICA
from params import *

findpeak =lambda f,w:round(f[np.argmax(w)],2)
def SM(x):return np.exp(x)/sum(np.exp(x))
confidence=lambda w:round(max(SM(w))*100,2)
bpm =lambda f,w:int(f[np.argmax(w)])

def ICA(rgb_signal,fps):
    mu=rgb_signal.mean(axis=0)
    sigma=rgb_signal.std(axis=0)
    normed_channel_avg=(rgb_signal-mu)/sigma
    transformer = FastICA(random_state=0)
    X_transformed = transformer.fit_transform(normed_channel_avg)
    

    fb,wb=FFT(X_transformed[:,0],fps)
    fg,wg=FFT(X_transformed[:,1],fps)
    fr,wr=FFT(X_transformed[:,2],fps)
    
    return {'ica_signals':X_transformed.T.tolist(),'c1':bpm(fb,wb),'c2':bpm(fg,wg),'c3':bpm(fr,wr),'Confidence':[fb,fg,fr,wb,wg,wr]}



    
def g_channel(channel,fps):
    f,w=FFT(channel,fps)
    return {'g_signals':channel.tolist(),'g':bpm(f,w),'gf':f,'gw':w}


def FFT(signal,fps,filter=True):
    w=np.fft.fft(signal)

    freq=np.fft.fftfreq(len(signal))*fps

    p=abs(w)
    if filter:
        mask=(freq<PASSBAND_HIGH) & (freq>PASSBAND_LOW)

        return (freq[mask]*60).tolist(),p[mask].tolist()
    return freq,p

