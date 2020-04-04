from distutils.util import strtobool
import os

resolution_name_map = {"HD":(1280,720), "VGA":(640,480), "13MP":(4160,3120), "8MP":(3264,2448)}
resolution_label_map = {"HD":(1280,720), "VGA":(1280, 960), "13MP":(1040,780), "8MP":(1040,780)} #this resolutions were used to make labelling easier
DS_path = "/home/maciej/Desktop/pallet_dataset/"
PROJECT_PATH = "/home/maciej/repos/pallet-recogntion-gpu/"
SCENES_PATH = DS_path+"scenes/"
IMG_SAFETY_BORDER=130
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


def get_scene_dir_path(scene):
    return DS_path+"scenes/"+scene+"/"