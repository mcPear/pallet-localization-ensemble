import cv2
import numpy as np

def get_channels_map():
    return {'B':0, 'G':1,'R':2,'H':3,'S':4,'V':5,'C':6,'M':7,'Y':8,'K':9,'L':10,'A':11,'b':12}

def bgr_to_cmyk(bgr):
    RGB_SCALE = 255
    CMYK_SCALE = 100
    b = bgr[0]
    g = bgr[1]
    r = bgr[2]
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, CMYK_SCALE

    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    return [c, m, y, k]

def to_rgb_channels(bgr, channels="HSCMYb"):
    bgr_umat = cv2.UMat(bgr)
    
    hsv = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2HSV)
    hsv=cv2.UMat.get(hsv)
    
    cmyk = np.apply_along_axis(bgr_to_cmyk, 2, bgr) #todo PIL?
    
    lab = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2LAB)
    lab=cv2.UMat.get(lab)
    
    f = np.concatenate((bgr, hsv, cmyk, lab),2)

    channels_map = get_channels_map()
    channels = list(channels)
    channels = [channels_map[ch] for ch in channels]
    f = f[:,:,channels]
    return f