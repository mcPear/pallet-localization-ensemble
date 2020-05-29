import cv2
import numpy as np
import cupy as cp
import time
import imutils
from joblib import load
from dataset_io import *
import timeit
from tqdm import tqdm
import psutil
from datetime import datetime
from profiler import *
import copy

# SET PARAMS
(winW, winH) = (WIN_W, WIN_H)
channels_no=8
DS_min_winH_m=41
DS_max_winH_m=281
initial_scale=winH/DS_min_winH_m
final_scale=winH/DS_max_winH_m
RESIZING_SCALE=1.15
MAX_PRED_OVERLAPPING=0.5
MIN_PRED=0.5
WHITE=(255, 255, 255)
DRAW_BORDER=4
KERNEL_W_RATIO=7.0/640.0 #because kernel (7,7) was the best for VGA images in previous research

h=winH/1.2
w=winW/1.2
x0=winW/12.0
y0=winH/12.0
x1=winW*11.0/12.0
y1=winH*11.0/12.0
((h1x1,h1y1),(h1x2,h1y2))=((int(0.126*w+x0), int(0.3056*h+y0)),(int(0.409*w+x0), int(h+y0)))
((h2x1,h2y1),(h2x2,h2y2))=((int(0.5906*w+x0), int(0.3056*h+y0)),(int(0.875*w+x0), int(h+y0)))

def pyramid(image, gradient_channels, binary_imgs, initial_scale, final_scale, scale):
    original_w=image.shape[1]
    curr_scale=initial_scale
    while curr_scale>final_scale/scale:
        w = int(original_w * curr_scale)
        yield imutils.resize(image, width=w), [imutils.resize(ch, width=w) for ch in gradient_channels], [imutils.resize(np.uint8(img), width=w).astype(bool) for img in binary_imgs],curr_scale
        curr_scale/=scale

def sliding_window(image, stepSize, windowSize):
    ignored_height_ratio=0.257 #calculated in dataset_processing.ipynb
    height=image.shape[0]
    start_height=int(ignored_height_ratio*image.shape[0])
    for y in range(start_height, height, stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def area(rect):
    ((x,y),(x2,y2))=rect
    return (x2-x)*(y2-y)

def calc_overlapping(a,b,method):
    dx = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])
    dy = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])
    basic_area=method([area(a), area(b)])
    overlap_area=dx*dy
    is_overlapping=((dx>=0) and (dy>=0))
    overlapping_ratio=overlap_area/basic_area

    return overlapping_ratio if is_overlapping else 0

def scale_many(vals, scale):
    return [int(val/scale) for val in vals]

def binarize(color_channels, color_clfs):
    s0, s1, s2 = color_channels[0].shape
    flat_ch = [np.reshape(img, (s0*s1, s2)) for img in color_channels]
    flat_img = np.hstack(flat_ch)
    flat_binaries=[clf.predict(flat_img) for clf in color_clfs]
    img_binaries=[np.reshape(flat, (s0, s1)) for flat in flat_binaries]
    return img_binaries

def open_bin(img, kernel):
    kernel = np.ones(kernel, np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    return img_dilation

def close_bin(img, kernel):
    kernel = np.ones(kernel, np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    return img_erosion

def morpho(img):
    h,w=img.shape
    k=int(w*KERNEL_W_RATIO)
    kernel=(k,k)
    return np.bool_(open_bin(close_bin(np.uint8(img), kernel), kernel))

def color_score(win):
    rect=win[int(y0):int(y1),int(x0):int(x1)]
    hole_1=win[h1y1:h1y2, h1x1:h1x2]
    hole_2=win[h2y1:h2y2, h2x1:h2x2]

    all_points=rect.shape[0]*rect.shape[1]
    hole_1_points=hole_1.shape[0]*hole_1.shape[1]
    hole_2_points=hole_2.shape[0]*hole_2.shape[1]

    hole_1_positives=np.count_nonzero(hole_1)
    hole_2_positives=np.count_nonzero(hole_2)
    rect_score=np.count_nonzero(rect)-hole_1_positives-hole_2_positives
    hole_1_score=hole_1_points-hole_1_positives
    hole_2_score=hole_2_points-hole_2_positives
    score=(rect_score+hole_1_score+hole_2_score)/all_points

#     print("full",all_points, hole_1_points, hole_2_points)
#     print("positives",rect_score, hole_1_positives, hole_2_positives)
#     print("scores",rect_score, hole_1_score, hole_2_score)
#     print(score)

#     for e in [win, rect, hole_1, hole_2]:
#         cv2.imshow("Window", np.float32(e))
#         cv2.waitKey(0)

    return score

def show(img, x, y, winW, winH, positives=[]):
    clone = np.float32(img.copy())
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), WHITE, DRAW_BORDER)
    for p in positives:
        x,y=p
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), WHITE, DRAW_BORDER)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)

def pred_overlapping(row, scaled_rect):
    x,y,w,h = scaled_rect
    a = ((x,y),(x+w,y+h))
    [scene_name, filename, scale, (rx,ry)] = row
    [rx,ry,rw,rh] = scale_many([rx,ry,winW,winH], scale)
    b = ((rx,ry),(rx+rw,ry+rh))

    overlap=calc_overlapping(a,b,np.min)
    return overlap

def remove_margin(rect):
    ((x1,y1),(x2,y2))=rect
    w,h=(abs(x1-x2), abs(y1-y2))
    center=((x1+x2)/2,(y1+y2)/2)
    w_margin=w/12
    h_margin=h/12
    x1+=w_margin
    x2-=w_margin
    y1+=h_margin
    y2-=h_margin
    return ((int(x1),int(y1)),(int(x2),int(y2)))

def max_aggregation(pred_color, pred_grad):
    return np.max([pred_color, pred_grad], 0)

def min_aggregation(pred_color, pred_grad):
    return np.min([pred_color, pred_grad], 0)

def sum_aggregation(pred_color, pred_grad):
    return np.sum([pred_color, pred_grad], 0)

def aggregations():
    return {"max_aggregation":max_aggregation, "min_aggregation":min_aggregation, "sum_aggregation":sum_aggregation}

class Algorithm:

    def __init__(self, train_scenes, test_scenes, fold_name):
        self.train_scenes = train_scenes
        self.test_scenes = test_scenes
        self.fold_name = fold_name
        self.glob_RES = np.empty((0,5), object)
        self.clf_grad = None
        self.color_clfs = None
        self.glob_overlappings = []

    def probs_to_preds(self,pred,IDX,rects_count, aggregation_name):
        greater_than_09=pred > MIN_PRED
        pred=pred[greater_than_09]
        IDX=IDX[greater_than_09]
        max_pred_ids=[]
        max_preds=[]
        max_pred_IDXs=[]
        for i in range(rects_count):
            if len(pred) >0:
                max_pred_id=np.argmax(pred)
                max_pred=pred[max_pred_id]
                max_preds.append(max_pred)
                max_pred_IDX=IDX[max_pred_id]
                max_pred_IDXs.append(max_pred_IDX)

                [scene_name, filename, scale, (x,y)]=max_pred_IDX
                scaled_rect=scale_many([x,y,winW,winH], scale)

                not_within_bools = np.array([pred_overlapping(row, scaled_rect) < MAX_PRED_OVERLAPPING for row in IDX])
                pred=pred[not_within_bools]
                IDX=IDX[not_within_bools]
            else:
                print("No more predictions with prob > {}".format(MIN_PRED))
        if len(max_preds)>0:
            max_preds=np.array([[e] for e in max_preds])
            res=np.append(max_pred_IDXs, max_preds, 1)
            self.glob_RES[aggregation_name]=np.vstack([self.glob_RES[aggregation_name],res])
        
    def image_predict(self,X,COL_SCORS,IDX,rects_count):
        if len(X)!=0 and len(COL_SCORS)!=0 and rects_count>0:
            pred_color=np.array([np.max(cs) for cs in COL_SCORS])
            pred_grad=np.array([p[1] for p in self.clf_grad.predict_proba(X)])    
            for agg_name,agg in agregations():
                pred = agg(pred_color, pred_grad)
                probs_to_preds(pred,copy.deepcopy(IDX),rects_count, agg_name)

    def predict(self, filename, scene_name, label_resolution, rects, *args):
        X=[]
        IDX=[]
        COL_SCORS=[]

        image=imread_resized(scene_name, filename, label_resolution)
        gradient_channels=read_split_channels(scene_name, filename, "channels_gradient")
        gradient_channels=[ch.astype('float32') for ch in gradient_channels] #if not, scikit will do it later slower
        color_channels=read_split_channels(scene_name, filename, "channels_color")
        binary_imgs=binarize(color_channels, self.color_clfs)
        binary_imgs=[morpho(img) for img in binary_imgs]
        for resized_img, resized_gradient_ch, resized_binary_imgs, scale in pyramid(image, gradient_channels, binary_imgs, initial_scale, final_scale, scale=RESIZING_SCALE):
            resized_ch=np.dstack(resized_gradient_ch)
            for (x, y, window) in sliding_window(resized_img, stepSize=4, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                channels_window=resized_ch[y:y + winH, x:x + winW]
                color_ch_windows=[img[y:y + winH, x:x + winW] for img in resized_binary_imgs]
                color_scores=[color_score(w) for w in color_ch_windows]
                COL_SCORS.append(color_scores)
                color_ch_windows=None
                X.append(channels_window)
                row=[scene_name, filename, scale, (x,y)]
                IDX.append(row)
                #show(resized_binary_imgs[0], x, y, winW, winH)
        X=np.reshape(X,(len(X),winW*winH*channels_no)) #cupy does not help
        self.image_predict(X, COL_SCORS, np.array(IDX), len(rects))

    ### CALCULATE OVERLAPPINGS
    def calc_overlappings(self, rects, pred_rects):  
        overlappings=[(np.max([calc_overlapping(pred_rect,rect, np.max) for pred_rect in pred_rects]) if len(pred_rects) > 0 else 0) for rect in rects]
        print(overlappings)
        self.glob_overlappings.extend(overlappings)

    def draw_predicted_rectangles(self, filename, scene_name, label_resolution, rects, *args):
        img=imread_resized(scene_name, filename, label_resolution)
        img_rows=self.glob_RES[np.where((self.glob_RES[:,0] == scene_name) * (self.glob_RES[:,1] == filename))]

        pred_rects=[]
    #     rects=[add_margin(correct_rect_ratio(rect)) for rect in rects]
        for max_row in img_rows:
            [scene_name, filename, scale, (x,y), pred]=max_row
            [x,y,winW_s,winH_s]=scale_many([x,y,winW,winH], scale)
            pred_rect = (x,y), (x + winW_s, y + winH_s)
            pred_rect = remove_margin(pred_rect)
            cv2.rectangle(img, pred_rect[0], pred_rect[1], (0, 255, 0), DRAW_BORDER)
            pred_rects.append(((x,y),(x+winW_s, y+winH_s)))
        for rect in rects: 
            ((x,y),(x2,y2))=rect
            cv2.rectangle(img, (x,y), (x2,y2), (255, 0, 0), DRAW_BORDER)

        save_image(img, scene_name, filename, "predicted_labels")
        print(filename)
        self.calc_overlappings(rects, pred_rects)
        
    def load_models(self): 
        self.clf_grad = load(MODELS_PATH+"rand_forest_clf_{}.joblib".format(self.fold_name)) 
        self.clf_grad.set_params(n_jobs=-1)

        self.color_clfs=[load(MODELS_PATH+'naive_bayes_clf_{}_{}.joblib'.format(color, self.fold_name)) for color in COLORS]

    def predict_scenes(self):
        [self.predict(*row) for row in tqdm([row for row in walk_dataset(self.test_scenes)])]
        
    def run(self):
        self.load_models()
        profiled('self.predict_scenes()', globals(), locals())
        [self.draw_predicted_rectangles(*row) for row in tqdm(walk_dataset(self.test_scenes))]
        mean_overlapping = np.mean(self.glob_overlappings)
        return mean_overlapping