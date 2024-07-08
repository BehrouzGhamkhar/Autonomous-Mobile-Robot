import os
from PIL import Image

for file in os.listdir():
    filename, extension  = os.path.splitext('/home/amol/ros2ws/src/Autonomous-Mobile-Robot/occupancy_grid_mapping/occupancy_grid_mapping/my_map.pgm')
    if extension == ".pgm":
        new_file = "{}.png".format(filename)
        with Image.open(file) as im:
            im.save(new_file)