from distutils.util import strtobool
import os
import cv2
import numpy as np

resolution_name_map = {"HD":(1280,720), "VGA":(640,480), "13MP":(4160,3120), "8MP":(3264,2448)}
resolution_label_map = {"HD":(1280,720), "VGA":(1280, 960), "13MP":(1040,780), "8MP":(1040,780)} #this resolutions were used to make labelling easier
DS_path = "/home/maciej/Desktop/pallet_dataset/"
PROJECT_PATH = "/home/maciej/repos/pallet-recogntion-gpu/"
SCENES_PATH = DS_path+"scenes/"
CHANNELS_PATH=DS_path+"processed_scenes/channels/"

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

def read_split_channels(scene_name, filename):
    filepath=get_precessed_filepath("channels", scene_name, filename)
    ch14_path=filepath.replace(".jpg", "_ch14.png")
    ch58_path=filepath.replace(".jpg", "_ch58.png")
    ch14 = cv2.imread(ch14_path, cv2.IMREAD_UNCHANGED)
    ch58 = cv2.imread(ch58_path, cv2.IMREAD_UNCHANGED)
    return [ch14,ch58]

def read_channels(scene_name, filename):
    split_channels=read_split_channels(scene_name, filename)
    return np.dstack(split_channels)

def walk_dataset():
    rows=[]
    scene_names = get_scene_names()
    for scene_name in scene_names:
        include,_,_,_,label_resolution=read_info(scene_name)
        if include:
            filenames = get_image_names(scene_name)
            labels_dict=read_labels_dict(scene_name)
            for filename in filenames:
                rects=labels_dict[filename]
                rows.append((filename, scene_name, label_resolution, rects))
    return rows