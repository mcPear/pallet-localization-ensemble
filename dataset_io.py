from distutils.util import strtobool
import os
import cv2
import numpy as np
from operator import itemgetter 
from itertools import groupby 
import random

resolution_name_map = {"HD":(1280,720), "VGA":(640,480), "13MP":(4160,3120), "8MP":(3264,2448)}
resolution_label_map = {"HD":(1280,720), "VGA":(1280, 960), "13MP":(1040,780), "8MP":(1040,780)} #this resolutions were used to make labelling easier
DS_path = "/home/maciej/Desktop/pallet_dataset/"
PROJECT_PATH = "/home/maciej/repos/pallet-recogntion-gpu/"
SCENES_PATH = DS_path+"scenes/"
CLF_DATASETS_PATH = DS_path+"clf_datasets/"
MODELS_PATH = PROJECT_PATH+"models/"
GRADIENT_CHANNELS_PATH=DS_path+"processed_scenes/gradient_channels/"
COLORS=['blue','dark','wooden']
WIN_H=22 #(18*1.2)
WIN_W=120

def get_info_path(scene_name):
    return get_scene_path(scene_name)+"info.txt"

def read_info(scene_name):
    info_path=get_info_path(scene_name)
    result={}
    with open(info_path) as f:
        for line in f:
           (key, val) = line.split()
           result[key] = val
    include = strtobool(result["INCLUDE"])
    light=result["LIGHT"] if include else None
    pallet_color = result["PALLET_COLOR"] if include else None
    resolution_name=result["RESOLUTION"] if include else None
    resolution = resolution_name_map[resolution_name] if include else None
    label_resolution = resolution_label_map[resolution_name] if include else None
    return include, light, pallet_color, resolution, label_resolution
    
def read_labels(scene):
    def parse_line(line):
        filename,points=line.split(" ", 1)
        point_list=points.split("] [")
        point_list=[points.strip('][ ').replace('.','').split(' ') for points in point_list]
        point_list=[[int(point) for point in points if point] for points in point_list]
        point_list=[(tuple(points[:2]),tuple(points[2:4])) for points in point_list]
        if point_list[0]==((),()):
            point_list=[]
        return filename,point_list

    filepath=get_scene_labels_path(scene)
    with open(filepath) as f:
        lines = [line.rstrip('\n') for line in f]
    lines = [parse_line(line) for line in lines]
    return lines

def get_filepath(scene_name, filename):
    return get_scene_path(scene_name)+filename

def get_precessed_filepath(output_dir_name, scene_name, filename):
    return get_processed_output_dir_path(output_dir_name)+scene_name+"/"+filename

def read_labels_dict(scene_name):
    labels=read_labels(scene_name)
    return {file : regions for file, regions in labels }

def get_scene_names():
    return sorted(os.listdir(SCENES_PATH))

def get_image_names(scene_name):
    scene_path=get_scene_path(scene_name)
    names = sorted(os.listdir(scene_path))
    return [f for f in names if f not in ["info.txt", "equalized", "denoised"]]

def get_scene_path(scene_name):
    return SCENES_PATH+scene_name+"/"

def get_scene_labels_path(scene):
    return DS_path+"labels/"+scene+".txt"

def get_processed_output_dir_path(output_dir_name):
    return DS_path+"processed_scenes/"+output_dir_name+"/"

def get_scene_dir_path(scene):
    return DS_path+"scenes/"+scene+"/"

def imread_resized(scene_name, filename, label_resolution):
    filepath=get_filepath(scene_name, filename)
    image = cv2.imread(filepath)
    return cv2.resize(image, label_resolution)

def read_split_channels(scene_name, filename, output_dir_name):
    filepath=get_precessed_filepath(output_dir_name, scene_name, filename)
    ch1_path=filepath.replace(".jpg", "_ch1.png")
    ch2_path=filepath.replace(".jpg", "_ch2.png")
    ch1 = cv2.imread(ch1_path, cv2.IMREAD_UNCHANGED)
    ch2 = cv2.imread(ch2_path, cv2.IMREAD_UNCHANGED)
    return [ch1,ch2]

def read_channels(scene_name, filename, output_dir_name):
    split_channels=read_split_channels(scene_name, filename)
    return np.dstack(split_channels)

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        None

def chdir(path, scene_name):
    mkdir(path)
    os.chdir(path) 
    mkdir(scene_name)
    os.chdir(scene_name) 

def save_image(img, scene_name, filename, output_dir_name, postfix=None, png=False, nested_out_dir=None):
    postfix="_"+str(postfix) if postfix is not None else ""
    filename=filename.replace(".jpg", postfix+".jpg")
    if png:
        filename=filename.replace(".jpg", ".png")
    output_dir_path = get_processed_output_dir_path(output_dir_name)
    nested_out_dir = nested_out_dir+"/" if nested_out_dir else ""
    output_dir_path = output_dir_path+nested_out_dir
    chdir(output_dir_path, scene_name)
    cv2.imwrite(filename, img)

def walk_dataset(scene_names=None):
    rows=[]
    scene_names = scene_names if scene_names else get_scene_names()
    for scene_name in scene_names:
        include,_,pallet_color,_,label_resolution=read_info(scene_name)
        if include:
            filenames = get_image_names(scene_name)
            labels_dict=read_labels_dict(scene_name)
            for filename in filenames:
                rects=labels_dict[filename]
                rows.append((filename, scene_name, label_resolution, rects, pallet_color))
    return rows

def add_margin(rect):
    ((x1,y1),(x2,y2))=rect
    w,h=(abs(x1-x2), abs(y1-y2))
    center=((x1+x2)/2,(y1+y2)/2)
    w_margin=0.2*w/2
    h_margin=0.2*h/2
    x1-=w_margin
    x2+=w_margin
    y1-=h_margin
    y2+=h_margin
    return ((int(x1),int(y1)),(int(x2),int(y2)))

def correct_rect_ratio(rect):
    ((x1,y1),(x2,y2))=rect
    w,h=(abs(x1-x2), abs(y1-y2))
    center=((x1+x2)/2,(y1+y2)/2)
    ratio=WIN_W/WIN_H
    h=w/ratio
    y1=int(center[1]-h/2)
    y2=int(center[1]+h/2)
    return ((x1,y1),(x2,y2))

def cross_validated_scenes(folds = 4):
    scene_names = get_scene_names()
    scenes = []
    for scene_name in scene_names:
        include,_,pallet_color,_,label_resolution=read_info(scene_name)
        pallet_color_group = 'blue' if pallet_color=='wooden,blue' else pallet_color
        if include:
            scenes.append((scene_name, pallet_color, pallet_color_group))

    def extend(arr, parts=folds):
        extended = False
        while len(arr) < parts:
            arr.extend(arr.copy())
            extended=True
        return arr[:parts] if extended else arr

    sorted_scenes = sorted(scenes, key=itemgetter(2))
    groups = [[x[:1] for x in v] for k, v in groupby(sorted_scenes, key=itemgetter(2))]
    [random.shuffle(group) for group in groups]
    groups = [extend(g) for g in groups]
    groups = [np.array_split(g,4) for g in groups]
    groups = np.transpose(groups)
    groups = [np.concatenate([np.array([j.tolist() for j in i]).flatten().tolist() for i in g]).tolist() for g in groups]
    
    folds=[]
   
    for i in range(len(groups)):
        group=groups[i]
        group_copy=groups.copy()
        del group_copy[i]
        group_copy = np.concatenate(group_copy).tolist()
        #print(len(groups), len(group_copy))
        fold=(group_copy, group, "fold_"+str(i))
        folds.append(fold)
    
    return folds

def get_existing_clf_ds_filepath(feature_name, scene, color=None):
    color_path=color+"_" if color else ""
    dir_path = CLF_DATASETS_PATH+feature_name+"/"
    file_path = dir_path+color_path+scene+".npy"
    mkdir(dir_path)
    return file_path