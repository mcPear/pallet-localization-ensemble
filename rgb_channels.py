import cv2
import numpy as np
from PIL import Image

def get_channels_map():
    return {'B':0, 'G':1,'R':2,'H':3,'S':4,'V':5,'C':6,'M':7,'Y':8,'K':9,'L':10,'A':11,'b':12}

def pil_bgr_to_cmyk(bgr_umat):
    img = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2RGB)
    img=cv2.UMat.get(img)
    img = Image.fromarray(img)
    img = img.convert('CMYK')
    return np.asarray(img)

def to_rgb_channels(bgr, channels="HSCMYb"):
    bgr_umat = cv2.UMat(bgr)
    
    hsv = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2HSV)
    hsv=cv2.UMat.get(hsv)
    
    cmyk = pil_bgr_to_cmyk(bgr_umat)
    
    lab = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2LAB)
    lab=cv2.UMat.get(lab)
    
    f = np.concatenate((bgr, hsv, cmyk, lab),2)

    channels_map = get_channels_map()
    channels = list(channels)
    channels = [channels_map[ch] for ch in channels]
    f = f[:,:,channels]
    return f