import os
import pandas as pd
import numpy as np
import time
import cv2
from dlib68 import getxy
import transplant
matlab = transplant.Matlab(jvm=False, desktop=False)
from limited import limits
from apexs import crop_face
import math
import matplotlib.pyplot as plt
import seaborn as sns
      

def drawpic(rads,name):        
    sns.distplot(rads)
    plt.savefig(str(name)+'gaosi.png',dpi=120,bbox_inches='tight')
    rads = sorted(rads)    
    test = pd.DataFrame(data=rads)
    test.to_csv(str(name)+'.csv')
    plt.plot(rads)
    plt.savefig(str(name)+'.png',dpi=120,bbox_inches='tight')

def return_rad(onset_path, apex_path):    
    sizea = 340
    onset_crop,apex_crop,sizea,onx,ony,apx,apy = crop_face(onset_path,apex_path,sizea)
    size = 320
    xl,xr,yl,yr = limits(onx,ony,size,sizea)
    onset_crop = onset_crop[yl:yr,xl:xr]
    xl,xr,yl,yr = limits(apx,apy,size,sizea)
    apex_crop = apex_crop[yl:yr,xl:xr]
    onset_crop = cv2.resize(onset_crop,(size,size))
    apex_crop = cv2.resize(apex_crop,(size,size))
    ox,oy = getxy(onset_crop,size)
    ox = np.array(ox)
    oy = np.array(oy)
    rad = matlab.maxflow(onset_crop,apex_crop,ox,oy,1)
    rads = rad
    print(rad)
    if rad <= 1:
        rad = round(math.exp(2.2/rad))
    elif rad < 8 and rad >1:
        rad = round(math.exp(2.1449-0.0309*rad**2-0.0064*rad))
    else:
        rad = 1
    if rad < 1:
        rad = 1
    elif rad >20:
        rad = 20
    print('p:',rad)    
    return rads
