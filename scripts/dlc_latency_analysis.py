'''
Refer to https://deeplabcut.github.io/DLC-inferencespeed-benchmark/ for benchmark
'''

import yaml
import dlclive
# from utils.utils import *
# from utils.helpers import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import argparse
import random
import os

def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument( '--project_root', type=str, required=True,
    #   help='Root directory of the project. (eg: "/home/<username>/Code/stim_behavior/")')
    # parser.add_argument( '--input_dir', type=str, required=True,
    #   help='Directory where the videos are stored. (eg: "/home/<username>/Data/processed/mouse-cshl")')
    # parser.add_argument( '--output_dir', type=str, required=True,
    #   help='Directory to store the prosvd data. (eg: "/home/<username>/Data/output/mouse-cshl")')
    return parser

def load_random_file(directory, seed=0):
    random.seed(seed)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        print("No files found in the specified directory.")
        return None
    random_file = random.choice(files)
    return os.path.join(directory, random_file)

# with open(f'../configs/latency.yaml', 'r') as file:
#     config = yaml.safe_load(file)
# root_dir = config['path']['root']
# video_dir = f"{root_dir}/{config['path']['video']}"
# model_path = f"{root_dir}/{config['path']['model']}"
# output_dir = f"{root_dir}/{config['path']['output']}"
video_filepath = "/home/sachinks/Data/raw/octopus/sample_videos/pinch_distal_220823_125213_000-1.mp4"
model_path = "/home/sachinks/Code/MyProjects/exported_models/octopus_Tent6_mobilenet_keypoint5"
output_dir = "/home/sachinks/Code/MyProjects/exported_models/eval"
print(video_filepath)
breakpoint()


dlclive.benchmark_videos(model_path, [video_filepath], output=output_dir, resize=[1])


def smooth_data(data, kernel_size = 20):
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = ndimage.convolve(data, kernel)
    return data_convolved


file_path = f'{output_dir}/benchmark_ceres_CPU_3.pickle'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

idx = 0 # only considering one video size at a time

inference_times = data['inference_times']
latency = inference_times[idx, :]
latency = latency[np.nonzero(latency)]
latency_fps = 1/latency

print("====================")
print("Latency")
print(f"In ms: {1e3*latency.mean():.2f} ± {1e3*latency.std():.2f} ms")
print(f"In fps: {latency_fps.mean():.1f} ± {latency_fps.std():.1f} fps")
print("====================")


print()
print("====================")
print("SYSTEM INFO")
print(f"OS: {data['op_sys']}")
print(f"{data['device_type']}: {data['device']}")
print("====================")

print()
print("====================")
print("VIDEO INFO")
filename = data['video_path'].split('\\')[-1]
print(f"File: {filename}")
video_size = data['im_size'][idx]
print(f"Video Frame Size: {video_size[0]}x{video_size[1]}")
print(f"FPS: {data['video_fps']}Hz")
print("====================")

print()
print(f"Model: {data['model_type']}")

latency_smoothed = smooth_data(latency)
X = range(len(latency))
plt.plot(latency_smoothed, linewidth=3, alpha=0.8)
plt.scatter(X, latency, s=2)
plt.plot()
plt.xlabel('Frame index')
plt.ylabel('Latency (s)')
plt.title(f"Video Frame Size: {video_size[0]}x{video_size[1]}")
# plt.show()
data.pop('inference_times')
# data['']
print(data)

