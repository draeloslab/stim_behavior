#!/usr/bin/env python3

import sys
import gi
import time
import argparse
import numpy as np
import cv2
import multiprocessing as mp
import queue
import psutil
import warnings
import os
from collections import deque

gi.require_version("Tcam", "1.0")
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Tcam, Gst, GLib

# Import DLCLive for GPU-accelerated pose estimation
try:
    from dlclive import DLCLive
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
except ImportError:
    warnings.warn("DLCLive not found. Live pose estimation will be disabled.")
    DLCLive = None

class TISCamera:
    def __init__(self, serial, width=640, height=480, framerate="30/1"):
        self.serial = serial
        self.width = width
        self.height = height
        self.framerate = framerate
        self.properties = []
        self.pipeline = None
        self.source = None
        self.sink = None

    def start_pipeline(self):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(
            f"tcambin name=source"
            f" ! video/x-raw,format=BGRx,width={self.width},height={self.height},framerate={self.framerate}"
            f" ! videoconvert ! appsink name=sink"
        )
        self.source = self.pipeline.get_by_name("source")
        self.sink = self.pipeline.get_by_name("sink")
        
        if self.serial:
            self.source.set_property("serial", self.serial)

        self.sink.set_property("emit-signals", True)
        self.sink.connect("new-sample", self.on_new_sample)

        self.apply_properties()
        
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop_pipeline(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

    def apply_properties(self):
        for prop in self.properties:
            try:
                self.source.set_tcam_property(prop['property'], prop['value'])
                print(f"Set {prop['property']} to {prop['value']}")
            except GLib.Error as err:
                print(f"Failed to set {prop['property']}: {err.message}")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")
            
            result, mapinfo = buf.map(Gst.MapFlags.READ)
            if result:
                image = np.ndarray(
                    (height, width, 4),
                    buffer=mapinfo.data,
                    dtype=np.uint8
                )
                buf.unmap(mapinfo)
                return image
        return None

def camera_capture_process(camera, frame_queue, stop_event):
    camera.start_pipeline()
    
    try:
        while not stop_event.is_set():
            frame = camera.on_new_sample(camera.sink)
            if frame is not None:
                try:
                    frame_queue.put_nowait((time.time(), frame))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    frame_queue.put_nowait((time.time(), frame))
    finally:
        camera.stop_pipeline()

def dlclive_process(config, frame_queue, result_queue, stop_event):
    if DLCLive is None or not config['use_dlc']:
        print("DLCLive not available or disabled. Skipping pose estimation.")
        return

    dlc_model = DLCLive(config['dlc_model_path'], tf_gpu=True)
    dlc_model.init_inference()

    last_inference_time = time.time()
    frame_buffer = deque(maxlen=5)  # Buffer to store frames for skipping

    while not stop_event.is_set():
        try:
            timestamp, frame = frame_queue.get(timeout=1)
            frame_buffer.append((timestamp, frame))

            current_time = time.time()
            if current_time - last_inference_time < config['min_inference_interval']:
                continue  # Skip this frame if not enough time has passed

            # Use the most recent frame for inference
            _, inference_frame = frame_buffer[-1]
            
            inference_start = time.time()
            pose = dlc_model.inference(inference_frame)
            inference_time = time.time() - inference_start

            if inference_time > config['max_inference_time']:
                print(f"Warning: Inference took {inference_time:.2f}s, longer than the maximum allowed time.")
                # Skip frames if inference took too long
                while len(frame_buffer) > 1:
                    frame_buffer.popleft()

            result_queue.put((timestamp, inference_frame, pose))
            last_inference_time = current_time

        except queue.Empty:
            continue

def save_predictions(predictions, output_dir, camera_serial):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{camera_serial}_predictions.csv")
    
    with open(filename, 'a') as f:
        for timestamp, pose in predictions:
            flattened_pose = pose.flatten()
            pose_str = ','.join(map(str, flattened_pose))
            f.write(f"{timestamp},{pose_str}\n")

def plot_keypoints(frame, pose, config):
    for i, (x, y) in enumerate(pose):
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def display_process(frame_queues, result_queues, video_writers, config, stop_event):
    windows = [f"Camera {i}" for i in range(len(frame_queues))]
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    predictions = {i: [] for i in range(len(frame_queues))}

    while not stop_event.is_set():
        for i, (frame_queue, result_queue, video_writer) in enumerate(zip(frame_queues, result_queues, video_writers)):
            try:
                if config['use_dlc'] and not result_queue.empty():
                    timestamp, frame, pose = result_queue.get_nowait()
                    predictions[i].append((timestamp, pose))
                elif not frame_queue.empty():
                    timestamp, frame = frame_queue.get_nowait()
                    pose = None
                else:
                    continue

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                if pose is not None:
                    frame_bgr = plot_keypoints(frame_bgr, pose, config)

                cv2.imshow(windows[i], frame_bgr)
                video_writer.write(frame_bgr)

            except queue.Empty:
                continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

    for i, predictions_list in predictions.items():
        save_predictions(predictions_list, config['output_dir'], f"camera_{i}")

    for window in windows:
        cv2.destroyWindow(window)

def list_devices():
    Gst.init(None)
    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Video/Source/tcam")
    
    devices = monitor.get_devices()
    if devices:
        print(f"{'Index':5}{'Model':20}\t{'Serial':15}\tIdentifier")
        for i, device in enumerate(devices):
            props = device.get_properties()
            print(f"{i:5}{props.get_string('model'):20}\t{props.get_string('serial'):15}\t{props.get_string('identifier')}")
        return devices
    else:
        print("No devices found.")
        return []

def select_cameras():
    devices = list_devices()
    if not devices:
        return []

    selected_indices = input("Enter the indices of the cameras you want to use (comma-separated): ")
    selected_indices = [int(idx.strip()) for idx in selected_indices.split(',')]

    selected_cameras = []
    for idx in selected_indices:
        if 0 <= idx < len(devices):
            props = devices[idx].get_properties()
            camera = TISCamera(props.get_string('serial'))
            selected_cameras.append(camera)
        else:
            print(f"Invalid index: {idx}")

    return selected_cameras

def set_camera_properties(camera):
    print(f"Setting properties for camera {camera.serial}")
    while True:
        property_name = input("Enter property name (or 'done' to finish): ")
        if property_name.lower() == 'done':
            break
        property_value = input(f"Enter value for {property_name}: ")
        camera.properties.append({"property": property_name, "value": property_value})

def check_system_resources():
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        raise MemoryError("System is low on memory. Cannot proceed with capture.")

    if psutil.cpu_percent(interval=1) > 90:
        raise ResourceWarning("CPU usage is very high. Performance may be impacted.")

def live_stream_all(args):
    check_system_resources()

    cameras = select_cameras()
    if not cameras:
        print("No cameras selected. Exiting.")
        return

    for camera in cameras:
        set_camera_properties(camera)

    config = {
        'use_dlc': args.use_dlc,
        'dlc_model_path': args.dlc_model_path if args.use_dlc else None,
        'output_dir': args.output_dir,
        'min_inference_interval': args.min_inference_interval,
        'max_inference_time': args.max_inference_time
    }

    stop_event = mp.Event()
    frame_queues = [mp.Queue(maxsize=30) for _ in cameras]
    result_queues = [mp.Queue(maxsize=30) for _ in cameras]

    capture_processes = []
    for camera, frame_queue in zip(cameras, frame_queues):
        p = mp.Process(target=camera_capture_process, args=(camera, frame_queue, stop_event))
        p.start()
        capture_processes.append(p)

    dlc_processes = []
    if config['use_dlc']:
        for frame_queue, result_queue in zip(frame_queues, result_queues):
            p = mp.Process(target=dlclive_process, args=(config, frame_queue, result_queue, stop_event))
            p.start()
            dlc_processes.append(p)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback to MP4V codec

    video_writers = [
        cv2.VideoWriter(
            f"output_{camera.serial}.mp4",
            fourcc,
            float(camera.framerate.split('/')[0]),
            (camera.width, camera.height)
        ) for camera in cameras
    ]

    display_process = mp.Process(target=display_process, args=(frame_queues, result_queues, video_writers, config, stop_event))
    display_process.start()

    try:
        display_process.join()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")
    finally:
        stop_event.set()
        for p in capture_processes + dlc_processes:
            p.join()
        for writer in video_writers:
            writer.release()

    print("All processes stopped. Videos saved.")

def main():
    parser = argparse.ArgumentParser(description="Advanced GPU-Optimized TIS Camera Control Script")
    parser.add_argument("command", choices=["list", "stream"], help="Command to execute")
    parser.add_argument("--use-dlc", action="store_true", help="Enable DLC live inference")
    parser.add_argument("--dlc-model-path", type=str, help="Path to the DLC model")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save predictions")
    parser.add_argument("--min-inference-interval", type=float, default=0.033, help="Minimum time between inferences")
    parser.add_argument("--max-inference-time", type=float, default=0.1, help="Maximum allowed time for inference")

    args = parser.parse_args()

    if args.command == "list":
        list_devices()
    elif args.command == "stream":
        live_stream_all(args)

if __name__ == "__main__":
    main()
