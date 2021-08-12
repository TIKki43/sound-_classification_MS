import cv2
import sys
sys.path.append('../')
import numpy as np
import pandas as pd

from PIL import Image, ImageFont, ImageDraw
sys.path.insert(1, '/home/timur/Documents/Projects/sound_classification/ag_files')
from ag_files.data_prep import classes

print(classes)