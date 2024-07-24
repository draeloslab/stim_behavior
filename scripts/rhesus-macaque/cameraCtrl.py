# NOTE: Current version used in the Chestek Lab rig, being rewritten under camera.py. Will be removed once camera.py is fully validated and tested
#!/usr/bin/env python3

# import filepath
import time
import threading
import os
import sys
import threading
import time
from enum import Enum
from typing import Optional
import sys
# import keyboard

import textwrap
import argparse
import imagingcontrol4 as ic4

# constants for output formatting
BOLD = "\033[1m"
END = "\033[0m"


class ImageType(Enum):
    BMP = "bmp"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    def __str__(self):
        return self.value

class VideoType(Enum):
    H264 = "h264"
    H265 = "h265"

    def __str__(self):
        return self.value

def list_devices():
    devs = ic4.DeviceEnum()
    dev_list = devs.devices()
    if any(dev_list):
        print(f"{'Model':20}\t{'Serial':8}\tIdentifier")
    for d in dev_list:
        print(f"{d.model_name:20}\t{d.serial:8}\t{d.unique_name}")

def list_properties(dev: ic4.DeviceInfo):
    grabber = ic4.Grabber()
    grabber.device_open(dev)
    for p in grabber.device_property_map.all:
        print(f"{p.name:<16} - {p.type} - {p.visibility}")
        print(f"Description: {p.description}")
        print(f"Available: {p.is_available} Locked: {p.is_locked} Readonly: {p.is_readonly}")
        print(f"Affects others: {p.is_selector}")
        # Additional property-specific details can be added here if needed
        print("")

def set_properties(dev: ic4.DeviceInfo, new_props: list[str]):
    grabber = ic4.Grabber()
    grabber.device_open(dev)
    props = grabber.device_property_map
    for p in new_props:
        name, value = p.split("=")
        props.set_value(name, value)
        print(name, value)

def save_image(dev: ic4.DeviceInfo, n_images: int, store_dir: str = os.getcwd(), file_name: str = "image-{C}", file_type: ImageType = ImageType.BMP):
    grabber = ic4.Grabber()
    grabber.device_open(dev)
    sink = ic4.SnapSink()
    grabber.stream_setup(sink)
    
    def save(f: ic4.ImageBuffer, f_name: str):
        if file_type == ImageType.BMP:
            f.save_as_bmp(f"{store_dir}/{f_name}.{file_type}")
        elif file_type == ImageType.JPEG:
            f.save_as_jpeg(f"{store_dir}/{f_name}.{file_type}")
        elif file_type == ImageType.PNG:
            f.save_as_png(f"{store_dir}/{f_name}.{file_type}")
        elif file_type == ImageType.TIFF:
            f.save_as_tiff(f"{store_dir}/{f_name}.{file_type}")

    images = sink.snap_sequence(n_images, 1000)
    for n in range(0, n_images):
        f_name = file_name.replace("{C}", str(n))
        save(images[n], f_name)

def save_video(dev: ic4.DeviceInfo, duration_s: int = 10, file_name: str = "video", video_type: VideoType = VideoType.H264):
    grabber = ic4.Grabber()
    grabber.device_open(dev)
    fps = grabber.device_property_map.find_float("AcquisitionFrameRate")
    writer = ic4.VideoWriter(ic4.VideoWriterType.MP4_H264)
    file_path = f"C:/Users/Jjmas/OneDrive/Desktop/Research/Anne/stim_behavior/{file_name}.mp4"
    
    def sink_connected(sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
        sink.alloc_and_queue_buffers(6)
        writer.begin_file(file_path, image_type, fps.value)
        return True
    
    def sink_disconnected(sink: ic4.QueueSink):
        writer.finish_file()

    def frames_queued(sink: ic4.QueueSink):
        while True:
            try:
                f = sink.pop_output_buffer()
                writer.add_frame(f)
            except ic4.IC4Exception:
                return

    listen = ic4.QueueSinkListener()
    listen.sink_connected = sink_connected
    listen.sink_disconnected = sink_disconnected
    listen.frames_queued = frames_queued
    sink = ic4.QueueSink(listen)
    grabber.stream_setup(sink)

    start_time = time.time()
    while time.time() - start_time < float(duration_s):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping recording due to key press.")
            break
        time.sleep(0.1)

    grabber.device_close()

def start_simultaneous_recording(duration_s: int = 10, video_type: VideoType = VideoType.H264, file_name: str = "video"):
    devs = ic4.DeviceEnum()
    devices = devs.devices()
    threads = []
    for i, dev in enumerate(devices):
        name = f"{file_name}_{i}"
        print(f"Starting recording for {dev.model_name} ({dev.serial}) to {name}.")
        t = threading.Thread(target=save_video, args=(dev, duration_s, name, video_type))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def get_device(serial: str) -> Optional[ic4.DeviceInfo]:
    dev_list = ic4.DeviceEnum.devices()
    for d in dev_list:
        if d.serial == serial:
            return d
    return None


def live_stream(dev: ic4.DeviceInfo):
    """
    Start a live stream.

    This will open a display window.
    """
    grabber = ic4.Grabber()
    grabber.device_open(dev)

    grabber.device_property_map.set_value(ic4.PropId.TRIGGER_MODE, "Off")

    display = ic4.FloatingDisplay()

    grabber.stream_setup(None, display)

    e = threading.Event()

    def window_closed(disp: ic4.Display):
        print("window closed called")
        e.set()

    cb_token = display.event_register_window_closed(window_closed)
    try:
        e.wait(timeout=None)
    except KeyboardInterrupt:
        pass

    display.event_remove_window_closed(cb_token)

def example_imagebuffer_numpy_opencv_live():
    class ProcessAndDisplayListener(ic4.QueueSinkListener):
    # Listener to demonstrate processing and displaying received images

        def __init__(self, d: ic4.Display):
            self.display = d

        def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
            # Just accept whatever is passed
            return True

        def frames_queued(self, sink: ic4.QueueSink):
            # Get the new buffer from the sink
            buffer = sink.pop_output_buffer()
            # Send the modified buffer to the display
            self.display.display_buffer(buffer)
    # Let the select a video capture device
    device_list = ic4.DeviceEnum.devices()
    for i, dev in enumerate(device_list):
        print(f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]")
    print(f"Select device [0..{len(device_list) - 1}]: ", end="")
    selected_index = int(input())
    dev_info = device_list[selected_index]

    # Open the selected device in a new Grabber
    grabber = ic4.Grabber()
    grabber.device_open(dev_info)

    # Create a floating display
    display = ic4.FloatingDisplay()
    display.set_render_position(ic4.DisplayRenderPosition.STRETCH_CENTER)

    # Configure an window-closed event handler that sets a flag when the window is closed
    exit_flag: bool = False

    def window_closed_handler(d: ic4.Display):
        nonlocal exit_flag
        exit_flag = True

    display.event_add_window_closed(window_closed_handler)

    # Create a listener to process and display the received images
    listener = ProcessAndDisplayListener(display)

    # Create a sink that passes the images to the listener
    # This sink fixes the pixel format to BGR8, so that the listener receives 3-channel color images
    # max_output_buffers is set to 1 so that we always get the latest available image
    sink = ic4.QueueSink(listener, [ic4.PixelFormat.BGR8], max_output_buffers=1)
    grabber.stream_setup(sink)

    # Wait for the window to be closed
    while not exit_flag:
        time.sleep(0.1)

    grabber.stream_stop()

    return frame_arrays

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List devices")

    props_parser = subparsers.add_parser("properties", help="List properties of device")
    props_parser.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    set_parser = subparsers.add_parser("set", help="Set properties of device")
    set_parser.add_argument("--properties", "-p", nargs="+", required=True, help="List of properties to be set. (e.g. -p a=1 b=2)")
    set_parser.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    image_parser = subparsers.add_parser("image", help="Save image from device")
    image_parser.add_argument("--count", "-c", help="Number of images to save. Default=1", default=1, type=int)
    image_parser.add_argument("--path", help="Directory in which to save images. Default=<current dir>", default=os.getcwd())
    image_parser.add_argument("--name", help="Filename of saved images", default="image-{C}")
    image_parser.add_argument("--type", help="Image types", type=ImageType, choices=list(ImageType), default=ImageType.BMP)
    image_parser.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    video_parser = subparsers.add_parser("video", help="Save video from device")
    video_parser.add_argument("--time", "-t", help="Time period that shall be saved in seconds. Default=10s", default=10, type=int)
    video_parser.add_argument("--name", help="Filename of saved videos", default="video")
    video_parser.add_argument("--type", help="Video types", type=VideoType, choices=list(VideoType), default=VideoType.H264)
    video_parser.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    sim_parser = subparsers.add_parser("sim", help="Start simultaneous recording of multiple devices")
    sim_parser.add_argument("--time", "-t", help="Time period that shall be saved in seconds. Default=10s", default=10, type=int)
    sim_parser.add_argument("--type", help="Video types", type=VideoType, choices=list(VideoType), default=VideoType.H264)
    sim_parser.add_argument("--name", help="Filename of saved videos", default="video")

    subparsers.add_parser("live", help="Display live stream.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        return 1

    ic4.Library.init()

    if args.command == "list":
        list_devices()
    elif args.command == "live":
        frame_arrays = live_stream()
        print(f"Captured {len(frame_arrays)} camera streams.")
        for i, frames in enumerate(frame_arrays):
            print(f"Camera {i}: {len(frames)} frames captured.")
    elif args.command == "sim":
        print('Press "q" to stop recording.')
        start_simultaneous_recording(args.time, args.type, args.name)
        print("Recording finished.")
    else:
        d = get_device(args.serial)
        if not d:
            print("Unable to find device with given serial!")
            return 1

        if args.command == "properties":
            list_properties(d)
        elif args.command == "set":
            set_properties(d, args.properties)
        elif args.command == "image":
            save_image(d, args.count, args.path, args.name, args.type)
        elif args.command == "video":
            save_video(d, args.time, args.name, args.type)
        else:
            parser.print_help()

    return 0

if __name__ == "__main__":
    sys.exit(main())