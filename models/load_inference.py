import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

'''
Refer to https://deeplabcut.github.io/DLC-inferencespeed-benchmark/ for benchmark
'''

def smooth_data(data, kernel_size = 20):
    kernel = np.ones(kernel_size) / kernel_size
    # data_convolved = np.convolve(data, kernel, mode='same')
    data_convolved = ndimage.convolve(data, kernel)
    return data_convolved


file_path = '/home/sachinks/Code/MyDLCLive/octopus/eval/benchmark_ceres_CPU_0.pickle'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

idx = 0 # only considering one video size at a time

inference_times = data['inference_times']
latency = inference_times[idx, :]
latency = latency[np.nonzero(latency)]
print("====================")
print("Latency")
print(f"In ms: {1e3*latency.mean():.2f} ± {1e3*latency.std():.2f} ms")
print(f"In fps: {1/latency.mean():.1f} ± {1/latency.std():.1f} fps")
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
# breakpoint()
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


# (Pdb) data.keys()
# dict_keys(['model', 'model_type', 'TFGPUinference', 'im_size', 'inference_times', 'stats', 'host_name', 'op_sys', 'python', 'device_type', 'device', 'freeze', 'python_version', 'git_hash', 'dlclive_version', 'video_path', 'video_codec', 'video_pixel_format', 'video_fps', 'video_total_frames', 'original_frame_size', 'dlclive_params'])