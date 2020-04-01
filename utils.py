from distutils.util import strtobool

resolution_name_map = {"HD":(1280,720), "VGA":(640,480), "13MP":(4160,3120), "8MP":(3264,2448)}
resolution_label_map = {"HD":(1280,720), "VGA":(1280, 960), "13MP":(1040,780), "8MP":(1040,780)} #this resolutions were used to make labelling easier
DS_path = "/home/maciej/Desktop/pallet_dataset/"
scenes_path = DS_path+"scenes/"

def read_info(scene_path):
    result={}
    with open(scene_path+"info.txt") as f:
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

def get_scene_labels_path(scene):
    return DS_path+"labels/"+scene+".txt"


def get_scene_dir_path(scene):
    return DS_path+"scenes/"+scene+"/"