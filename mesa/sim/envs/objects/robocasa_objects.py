import os
import re
from collections import defaultdict

from mesa.sim import ASSETS_ROOT
from mesa.sim.envs.objects.utils import get_obj_info

from .base_object import (
    BaseObject,
    register_graspable_object,
    register_object,
    register_on_object,
    register_stackable_object,
)

BASE_ASSET_PATH = os.path.join(ASSETS_ROOT, "objects", "robocasa")
BASE_AI_ASSET_PATH = os.path.join(ASSETS_ROOT, "objects", "robocasa_ai")

SCALE_MAP = defaultdict(lambda: None)
SCALE_MAP.update({
    "bell_pepper_1": 0.9, # makes it much easier to grasp
    "bar_0": 0.8, # The original asset is just a bit too big
    "bar_2": 0.6, # The original asset is an absurdly large snickers bar
    "bar_3": 0.8, # The original asset is just a bit too big
    "bar_4": 0.8, # The original asset is just a bit too big
    "bar_6": 0.8, # The original asset is just a bit too big
    "bar_7": 0.8, # The original asset is just a bit too big
    "bar_8": 0.8, # The original asset is just a bit too big
    "bar_9": 0.8, # The original asset is just a bit too big
    "bar_10": 0.8, # The original asset is just a bit too big
    "bar_11": 0.8, # The original asset is just a bit too big
})

SCALE_MAP_AI = defaultdict(lambda: None)
SCALE_MAP_AI.update({
    "baking_sheet_2": 1.4, # The original asset is just a bit too small
    "tray_1": 1.6, # The original asset is just a bit too small
    "tray_4": 1.6, # The original asset is just a bit too small
    "tray_5": 1.6, # The original asset is just a bit too small
    "tray_6": 1.6, # The original asset is just a bit too small
    "tray_9": 1.6, # The original asset is just a bit too small
    "cutting_board_0": 1.8,
    "cutting_board_1": 1.8,
    "cutting_board_2": 1.8,
    "cutting_board_3": 1.8,
    "cutting_board_4": 1.8,
    "cutting_board_5": 1.8,
    "cutting_board_6": 1.8,
    "cutting_board_7": 1.8,
    "cutting_board_8": 1.8,
    "cutting_board_10": 1.8,
    "pot": 2.25,
    "squash_0": 0.7,
    "beet_2": 0.8,
})

class RobocasaObject(BaseObject):
    def __init__(
        self, 
        relative_path, 
        name, 
        joints=[dict(type="free", damping="0.0005")],
        scale=None,
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, relative_path),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=True,
            scale=scale,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z]|\d+)", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = None
        self.rotation_axis = "x"

        articulation_object_properties = {
            "default_open_ranges": [],
            "default_close_ranges": [],
        }
        self.object_properties = {
            "articulation": articulation_object_properties,
            "vis_site_names": {},
        }
    

# This code creates and registers all the robocasa objects
# List all directories up to 3 levels deep
asset_dirs = []
for root, dirs, _ in os.walk(BASE_ASSET_PATH):
    depth = root[len(BASE_ASSET_PATH):].count(os.sep)
    if depth == 1:
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            rel_path = os.path.relpath(full_path, BASE_ASSET_PATH)
            asset_dirs.append(rel_path)

ai_asset_dirs = []
for root, dirs, _ in os.walk(BASE_AI_ASSET_PATH):
    depth = root[len(BASE_AI_ASSET_PATH):].count(os.sep)
    if depth == 1:
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            rel_path = os.path.relpath(full_path, BASE_AI_ASSET_PATH)
            ai_asset_dirs.append(rel_path)


# Use this to make sure each object refers to the correct info
def make_init(xml_path, default_name, base_cls, scale):
    def __init__(self, *args, name=default_name, joints=None, scale=scale, **kwargs):
        if joints is None:
            joints = [dict(type="free")]
        # Call the base class __init__ with bound defaults, allow overrides
        base_cls.__init__(self, xml_path, *args, name=name, joints=joints, scale=scale, **kwargs)
    return __init__

def register_robocasa_object(asset_dir, last, is_ai=False):
    last = asset_dir.split(os.sep)[-1]

    obj_info = get_obj_info(f'robocasa_{last}')


    if is_ai:
        # if last in BLACKLISTED_OBJECTS_AI:
        #     return
        
        scale = SCALE_MAP_AI[last]
        if scale is None and 'scale' in obj_info['aigen']:
            scale = obj_info['aigen']['scale']
        
        
        class_name = f"RobocasaAI{last.title().replace('_', '')}"
        default_name = f"robocasa_ai_{last}"
        xml_path = os.path.join(BASE_AI_ASSET_PATH, asset_dir, "model.xml")
    else:
        if 'exclude' in obj_info['objaverse'] and \
            last in obj_info['objaverse']['exclude']: # or \
            # last in BLACKLISTED_OBJECTS:    
            return
        
        scale = SCALE_MAP[last]
        
        class_name = f"Robocasa{last.title().replace('_', '')}"
        default_name = f"robocasa_{last}"
        xml_path = os.path.join(BASE_ASSET_PATH, asset_dir, "model.xml")

    cls = type(
        class_name,
        (RobocasaObject,),
        {
            "__init__": make_init(xml_path, default_name, RobocasaObject, scale=scale),
            "__module__": __name__,  # nice for pickling/introspection
        },
    )

    register_object(cls)
    if obj_info["graspable"]:
        register_graspable_object(cls)
    if obj_info["on_dest"]:
        register_on_object(cls)
    if obj_info["types"] and "stackable" in obj_info["types"]:
        xy_threshold = obj_info["stack_xy_threshold"] if "stack_xy_threshold" in obj_info else 0.03
        z_threshold = obj_info["stack_z_threshold"] if "stack_z_threshold" in obj_info else 0.07
        register_stackable_object(cls, xy_threshold=xy_threshold, z_threshold=z_threshold)

for i, asset_dir in enumerate(asset_dirs):
    register_robocasa_object(asset_dir, i)

for i, asset_dir in enumerate(ai_asset_dirs):
    register_robocasa_object(asset_dir, i, is_ai=True)
