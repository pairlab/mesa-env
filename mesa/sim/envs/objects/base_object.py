import re

import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import string_to_array

OBJECTS_DICT = {}
VISUAL_CHANGE_OBJECTS_DICT = {}
GRASPABLE_OBJECTS_DICT = {}

# These are destination objects. 
# Objects that other objects can be put into. (eg basket)
IN_OBJECTS_DICT = {}
# Objects that other objects can be placed on. (eg plate)
ON_OBJECTS_DICT = {}
# any fixture such as table, drawer, includes articulated objs
FIXTURES_DICT = {}

STACKABLE_OBJECTS_DICT = {}

def smart_camel_to_snake(name):
    # Insert underscores before capitals (excluding consecutive capitals for acronyms) and before digits
    s1 = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    s3 = re.sub('([a-zA-Z])([0-9])', r'\1_\2', s2)
    return s3.lower()

def register_stackable_object(target_class, xy_threshold=0.03, z_threshold=0.07):
    """We keep track of objects that can be stacked. We need to know the xy and z thresholds for stacking."""
    # Modified to not add underscore between consecutive capital letters (i.e., acronyms like "AI")
    key = smart_camel_to_snake(target_class.__name__)
    STACKABLE_OBJECTS_DICT[key] = {
        "class": target_class,
        "z_threshold": z_threshold,
        "xy_threshold": xy_threshold,
    }
    return target_class

def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT

def register_object(target_class):
    """We design the mapping to be case-INsensitive."""
    key = smart_camel_to_snake(target_class.__name__)
    assert key not in OBJECTS_DICT
    OBJECTS_DICT[key] = target_class
    return target_class


def register_visual_change_object(target_class):
    """We keep track of objects that might have visual changes to optimize the codebase"""
    key = smart_camel_to_snake(target_class.__name__)
    VISUAL_CHANGE_OBJECTS_DICT[key] = target_class
    return target_class


def register_graspable_object(target_class):
    """We keep track of objects that can be grasped"""
    key = smart_camel_to_snake(target_class.__name__)
    GRASPABLE_OBJECTS_DICT[key] = target_class
    return target_class


def register_in_object(regions):
    """Decorator factory that registers a class in IN_OBJECTS_DICT."""
    def decorator(target_class):
        key = smart_camel_to_snake(target_class.__name__)
        IN_OBJECTS_DICT[key] = {
            "class": target_class,
            "regions": regions,
        }
        return target_class
    return decorator


def register_on_object(target_class):
    """We keep track of objects that can be placed on"""
    key = smart_camel_to_snake(target_class.__name__)
    ON_OBJECTS_DICT[key] = target_class
    return target_class

def register_fixture(target_class):
    """We keep track of objects that are fixtures, e.g. drawers, tables..."""
    key = smart_camel_to_snake(target_class.__name__)
    FIXTURES_DICT[key] = target_class
    return target_class


class BaseObject(MujocoXMLObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rotation_axis = "x"

    @property
    def horizontal_offset(self) -> np.ndarray:
        """
        Returns the offset of the horizontal radius site.
        """
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        return string_to_array(horizontal_radius_site.get("pos"))

    @property
    def horizontal_radius(self) -> float:
        return np.linalg.norm(self.horizontal_offset)