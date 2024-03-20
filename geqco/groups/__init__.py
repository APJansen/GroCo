from .group import *
from .space_groups import space_group_dict
from .wallpaper_groups import wallpaper_group_dict

group_dict = {**wallpaper_group_dict, **space_group_dict}
