import os
from proSVD import proSVD
import numpy as np
import matplotlib.pyplot as plt
from stim_behavior.utils.utils import *
from stim_behavior.data_manager import DataManager
from stim_behavior.mice_data_loader import MiceDataLoader
from stim_behavior.prosvd_manager import ProSVDManager

def main(video_dir, output_dir):
    dim_k = 4
    stop_at = 150

    prosvdmanager = ProSVDManager(dim_k = dim_k, stop_at = stop_at)

    video_filenames = []
    for item in os.listdir(video_dir):
        item_path = os.path.join(video_dir, item)
        if os.path.isfile(item_path):
            video_filenames.append(item)

    for filename in video_filenames:
        video_path = os.path.join(video_dir, filename)
        data, metadata = prosvdmanager.process_video(video_path = video_path)
        output_video_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        prosvdmanager.save_result(data, output_dir=output_video_dir)    

if __name__ == "__main__":
    video_dir = "/home/sachinks/Data/processed/mouse-cshl/sample_videos"
    output_dir = "/home/sachinks/Data/tmp"
    main(video_dir, output_dir)