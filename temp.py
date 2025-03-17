import os
import warnings
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

# Get the current working directory
root = os.getcwd()
path_to_coco = os.path.join(root, "xtcocoapi")
print(path_to_coco)