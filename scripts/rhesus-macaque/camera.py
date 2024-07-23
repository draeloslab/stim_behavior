#!/usr/bin/env python3

import time
import threading
import os
from enum import Enum
import sys
import keyboard
import argparse
import imagingcontrol4 as ic4
import numpy as np
from dlclive import DLCLive

class ImageType(Enum):
    """Supported image types for saving."""
    BMP = "bmp"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

class VideoType(Enum):
    """Supported video types for saving."""
    H264 = "h264"
    H265 = "h265"

def list_devices():
    """List available camera devices."""
    devices = ic4.DeviceEnum().devices()
    if devices:
        print(f"{'Model':20}\t{'Serial':8}\tIdentifier")
        for dev in devices:
            print(f"{dev.model_name:20}\t{dev.serial:8}\t{dev.unique_name}")
    else:
        print("No devices found.")

def get_device(serial: str):
    """Get device info for a given serial number."""
    for dev in ic4.DeviceEnum().devices():
        if dev.serial == serial:
            return dev
    return None

def set_properties(grabber: ic4.Grabber, properties: list[str]):
    """Set camera properties."""
    props = grabber.device_property_map
    for prop in properties:
        name, value = prop.split("=")
        props.set_value(name, value)
        print(f"Set {name} to {value}")

class CameraStream:
    def __init__(self, dev: ic4.DeviceInfo, properties: list[str] = None):
        self.dev = dev
        self.grabber = ic4.Grabber()
        self.display = ic4.FloatingDisplay()
        self.running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        self.properties = properties
        self.dlclive = DLCLive('/home/jakejoseph/Desktop/Joseph_Code/CentralNHPTracker-Jake-2024-07-19/exported-models/DLC_CentralNHPTracker_mobilenet_v2_1.0_iteration-0_shuffle-1')
        # TODO: Add DLC model path as an argument or in config file


    def start(self):
        self.grabber.device_open(self.dev)
        self.grabber.device_property_map.set_value(ic4.PropId.TRIGGER_MODE, "Off")
        
        if self.properties:
            set_properties(self.grabber, self.properties)

        class ProcessAndDisplayListener(ic4.QueueSinkListener):
            def __init__(self, display: ic4.Display, parent: 'CameraStream'):
                self.display = display
                self.parent = parent

            def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
                return True

            def frames_queued(self, sink: ic4.QueueSink):
                buffer = sink.pop_output_buffer()
                self.display.display_buffer(buffer)
                with self.parent.frame_lock:
                    self.parent.latest_frame = buffer.numpy_copy()
                    self.parent.frame_count += 1

        listener = ProcessAndDisplayListener(self.display, self)
        sink = ic4.QueueSink(listener, [ic4.PixelFormat.BGR8], max_output_buffers=1)
        self.grabber.stream_setup(sink)

        self.display.set_window_title(f"{self.dev.model_name} ({self.dev.serial})")

        while self.running:
            time.sleep(0.01)

        self.grabber.stream_stop()
        self.grabber.device_close()

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_frame_count(self):
        with self.frame_lock:
            return self.frame_count

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0
    
    def run_inference(self, frame: np.ndarray):
        self.dlclive.init_inference(frame)
        return self.dlclive.get_pose(frame)

def live_stream_all(properties: list[str] = None):
    """Start a live stream display for all available devices and return frames."""
    devices = ic4.DeviceEnum().devices()
    if not devices:
        print("No devices found.")
        return

    streams = [CameraStream(dev, properties) for dev in devices]
    threads = [threading.Thread(target=stream.start) for stream in streams]

    for thread in threads:
        thread.start()

    print("Press 'q' to stop all live streams.")
    try:
        while not keyboard.is_pressed('q'):
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
            print("Live Stream Statistics:")
            print("----------------------")
            for i, stream in enumerate(streams):
                frame = stream.get_latest_frame()
                prediction = stream.run_inference(frame) # TODO: Might decide to move this into a separate process or directly in the Image Buffer
                frame_count = stream.get_frame_count()
                fps = stream.get_fps()
                if frame is not None:
                    print(f"Camera {i} ({stream.dev.serial}):")
                    print(f"  Frame shape: {frame.shape}")
                    print(f"  Frames processed: {frame_count}")
                    print(f"  FPS: {fps:.2f}")
                    print("----------------------")
            time.sleep(0.1)  # Adjust this delay as needed
    finally:
        for stream in streams:
            stream.stop()

        for thread in threads:
            thread.join()

def main():
    """Main function to handle command-line arguments and execute camera operations."""
    parser = argparse.ArgumentParser(description="Camera Control Script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List devices
    subparsers.add_parser("list", help="List available devices")

    # Live stream (all cameras)
    live_parser = subparsers.add_parser("live", help="Display live stream from all available devices and extract frames")
    live_parser.add_argument("--properties", "-p", nargs="+", help="List of properties to be set. (e.g. -p Width=1920 Height=1080 AcquisitionFrameRate=30)")

    args = parser.parse_args()

    ic4.Library.init()

    try:
        if args.command == "list":
            list_devices()
        elif args.command == "live":
            live_stream_all(args.properties)
    finally:
        ic4.Library.exit()

if __name__ == "__main__":
    main()