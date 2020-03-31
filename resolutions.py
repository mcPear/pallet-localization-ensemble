resolution_name_map = {"HD":(1280,720), "VGA":(640,480), "13MP":(4160,3120), "8MP":(3264,2448)}
resolution_label_map = {"HD":(1280,720), "VGA":(1280, 960), "13MP":(1040,780), "8MP":(1040,780)} #this resolutions were used to make labelling easier

def read_info(scene_path):
    result={}
    with open(scene_path+"info.txt") as f:
        for line in f:
           (key, val) = line.split()
           result[key] = val
    include = result["INCLUDE"]
    light=result["LIGHT"]
    pallet_color = result["PALLET_COLOR"]
    resolution_name=result["RESOLUTION"]
    resolution = resolution_name_map[resolution_name]
    label_resolution = resolution_label_map[resolution_name]
    return include, light, pallet_color, resolution, label_resolution
    