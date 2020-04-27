import cv2
import numpy as np
from PIL import Image, ImageCms
from dataset_io import PROJECT_PATH

def get_channels_map():
    return {'B':0, 'G':1,'R':2,'H':3,'S':4,'V':5,'C':6,'M':7,'Y':8,'K':9,'L':10,'A':11,'b':12}

def pil_bgr_to_cmyk(bgr_umat):
    img = cv2.cvtColor(bgr_umat, cv2.COLOR_BGR2RGB)
    img=cv2.UMat.get(img)
    img = Image.fromarray(img)
    img = ImageCms.profileToProfile(img, PROJECT_PATH+'color_profiles/AdobeRGB1998.icc', PROJECT_PATH+'color_profiles/USWebUncoated.icc', outputMode='CMYK')
    return np.asarray(img)

def equalize_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2])
    return cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)

def to_color_channels(bgr, channels="HSCMYb"):
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
    
    for i in range(len(channels)):
        f[:,:,i] = cv2.equalizeHist(f[:,:,i])
    
    return f