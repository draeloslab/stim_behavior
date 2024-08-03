import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from PIL import Image
import yaml
import pickle
import json
from tqdm import tqdm

from proSVD import proSVD
# from dlclive import DLCLive, Processor

from stim_behavior.utils.utils import *
from stim_behavior.utils.helpers import *


def main(path_dict):
    video_dir = path_dict['video_dir']
    output_dir = path_dict['output_dir']
    metadata_path = path_dict['metadata_path']
    model_path = path_dict['model_path']

    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)

    fps = metadata['fps']
    filenames_from_metadata = [fileinfo["file_name"] for fileinfo in metadata['video_data']]
    filenames_from_videodir = [os.path.splitext(file)[0] for file in os.listdir(video_dir)]
    filenames = list(set(filenames_from_metadata) & set(filenames_from_videodir))

    working_dir = output_dir # to save processed data and figures
    model_path = model_path

    # dlc_live = DLCLive(
    #     model_path,
    #     processor=Processor(),
    #     pcutoff=0.2,
    #     resize=1)

    PROSVD_K = 4 # no. of dims to reduce to

    TIME_MARGIN = (-120, 180) # to trim videos
    total_f = TIME_MARGIN[1] - TIME_MARGIN[0]

    init_frame_crop = 10 # No of initial frames used to set cropping info
    init_frame_prosvd = 90 # No of initial frames used to initialize proSVD
    init_frame = init_frame_crop + init_frame_prosvd

    print(f"Processing {len(filenames)} videos from {video_dir}")
    if len(filenames) < 4:
        print('\t', end='')
        print(*filenames, sep="\n\t")

if __name__ == "__main__":
    path_dict = {
        'metadata_path': "/home/sachinks/Data/raw/octopus/metadata.json",
        'video_dir': "/home/sachinks/Data/raw/octopus/sample_videos",
        'output_dir': "/home/sachinks/Data/tmp",
        'model_path': "/home/sachinks/Code/MyProjects/exported_models/DLC_Octo-Arm_resnet_50_iteration-0_shuffle-1-keypoint17"
    }
    main(path_dict, "/home/sachinks/Data/raw/octopus/metadata.xlsx")
