#!/usr/bin/env python3

# import filepath
import time
import threading
import os
from enum import Enum
from typing import Optional
import sys
import keyboard

import textwrap
import argparse
import imagingcontrol4 as ic4

# constants for output formatting
BOLD = "\033[1m"
END = "\033[0m"


class ImageType(Enum):
    """
    Helper enum for easier argparse generation.
    Contains supported image types.
    """

    BMP = "bmp"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"

    def __str__(self):
        return self.value


class VideoType(Enum):
    """
    Helper enum for easier argparse generation.
    Contains supported image types.
    """

    H264 = "h264"
    H265 = "h265"

    def __str__(self):
        return self.value


def list_devices():
    """
    List available devices
    """
    devs = ic4.DeviceEnum()

    dev_list = devs.devices()

    if any(dev_list):
        print(f"{'Model':20}\t{'Serial':8}\tIdentifier")

    for d in dev_list:
        print(f"{d.model_name:20}\t{d.serial:8}\t{d.unique_name}")


def list_properties(dev: ic4.DeviceInfo):
    """
    List properties of device
    """

    grabber = ic4.Grabber()
    grabber.device_open(dev)

    for p in grabber.device_property_map.all:
        print(f"{BOLD}{p.name:<16}{END} - {p.type} - {p.visibility}")
        print(textwrap.fill(f"{p.description}", 80, initial_indent="\t", subsequent_indent="\t"))
        print("")
        print(f"\tAvailable: {p.is_available} Locked: {p.is_locked} Readonly: {p.is_readonly}")
        print(f"\tAffects others: {p.is_selector}")
        # if p.is_selector:
        victims = p.selected_properties
        if victims:
            for v in victims:
                print(f"\t\t{v.name}")
            print()

        if isinstance(p, ic4.PropBoolean):
            if p.has_default:
                print(f"\tDefault: {p.default}")

        elif isinstance(p, ic4.PropInteger):
            print(f"\tRepresentation: {p.representation}")
            print(f"\tRange: {p.minimum} - {p.maximum}")
            print(f"\tIncrment mode: {p.increment_mode}")
            if p.increment_mode == ic4.PropertyIncrementMode.INCREMENT:
                print(f"\tIncrement: {p.increment}")
            elif p.increment_mode == ic4.PropertyIncrementMode.VALUE_SET:
                print(f"\tValidValueSet: {p.valid_value_set}")

        elif isinstance(p, ic4.PropFloat):
            print(f"\tRepresentation: {p.representation}")
            print(f"\tRange: {p.minimum} - {p.maximum}")
            print(f"\tIncrment mode: {p.increment_mode}")
            if p.increment_mode == ic4.PropertyIncrementMode.INCREMENT:
                print(f"\tIncrement: {p.increment}")
            elif p.increment_mode == ic4.PropertyIncrementMode.VALUE_SET:
                print(f"\tValidValueSet: {p.valid_value_set}")

        elif isinstance(p, ic4.PropEnumeration):
            e = p.entries
            print("\tAvailable entries:")
            for entry in e:
                print(f"\t{entry.value} - {entry.name}")

        if (
            isinstance(p, ic4.PropInteger)
            or isinstance(p, ic4.PropBoolean)
            or isinstance(p, ic4.PropFloat)
            or isinstance(p, ic4.PropEnumeration)
            or isinstance(p, ic4.PropString)
            or isinstance(p, ic4.PropRegister)
        ):
            try:
                value = p.value
            except ic4.IC4Exception as e:
                value = "==Unknown=="
                if e.code == ic4.Error.GenICamChunkdataNotConnected:
                    print("Requires ChunkData to be connected")
            except AttributeError as e:
                print(f"{p.name} {e}")
                continue
            print("")
            print(f"\tValue: {value}")

        print("")  # empty line just for readability


def set_properties(dev: ic4.DeviceInfo, new_props: list[str]):
    """"""

    grabber = ic4.Grabber()
    grabber.device_open(dev)

    props = grabber.device_property_map

    for p in new_props:
        name, value = p.split("=")

        props.set_value(name, value)
        print(name, value)

def save_image(
    dev: ic4.DeviceInfo,
    n_images: int,
    store_dir: str = os.getcwd(),
    file_name: str = "image-{C}",
    file_type: ImageType = ImageType.BMP,
):
    """
    Save video to file.

    Args:
        dev: Device that shall be opened
        store_dir: Default: current working dir
        file_name: Default: "video"
        video_type: Default: bmp
    """
    grabber = ic4.Grabber()
    grabber.device_open(dev)

    sink = ic4.SnapSink()

    grabber.stream_setup(sink)

    def save(f: ic4.ImageBuffer, f_name: str):
        # error via exception
        if file_type == ImageType.BMP:
            f.save_as_bmp(f"{store_dir}/{f_name}.{file_type}")
        elif file_type == ImageType.JPEG:
            f.save_as_jpeg(f"{store_dir}/{f_name}.{file_type}")
        if file_type == ImageType.PNG:
            f.save_as_png(f"{store_dir}/{f_name}.{file_type}")
        if file_type == ImageType.TIFF:
            f.save_as_tiff(f"{store_dir}/{f_name}.{file_type}")

    images = sink.snap_sequence(n_images, 1000)
    for n in range(0, n_images):
        f_name = file_name.replace("{C}", str(n))
        save(images[n], f_name)

def save_video(
    dev: ic4.DeviceInfo, duration_s: int = 10, file_name: str = "video", video_type: VideoType = VideoType.H264
):
    """
    Save video to file.

    Args:
        dev: Device that shall be opened
        duration_s: Default: 10
        file_name: Default: "video"
        video_type: Default: h264
    """
    grabber = ic4.Grabber()
    grabber.device_open(dev)

    fps = grabber.device_property_map.find_float("AcquisitionFrameRate")

    writer = ic4.VideoWriter(ic4.VideoWriterType.MP4_H264)

    class Listener(ic4.QueueSinkListener):
        def __init__(self, file_path):
            self.file_path = file_path

        def sink_connected(self, sink: ic4.QueueSink, image_type: ic4.ImageType, min_buffers_required: int) -> bool:
            sink.alloc_and_queue_buffers(6)
            writer.begin_file(self.file_path, image_type, fps.value)
            return True
        
        def sink_disconnected(self, sink: ic4.QueueSink):
            writer.finish_file()

        def frames_queued(self, sink: ic4.QueueSink):
            while True:
                try:
                    f = sink.pop_output_buffer()
                    writer.add_frame(f)
                except ic4.IC4Exception:
                    return
    file_path = f"C:/Users/Jjmas/OneDrive/Desktop/Research/Anne/stim_behavior/{file_name}.mp4"
    listen = Listener(file_path)
    sink = ic4.QueueSink(listen)

    grabber.stream_setup(sink)


    #Added functionality to stop recording earlier if necessary
    start_time = time.time()
    while time.time() - start_time < float(duration_s):
        if keyboard.is_pressed('q'):  # If 'q' is pressed, stop recording
            print("Stopping recording due to key press.")
            break
        time.sleep(0.1)  # Sleep briefly to reduce CPU usage

    grabber.device_close()  # Ensure the device is properly closed

    # time.sleep(duration_s)




def threaded_video_capture(dev: ic4.DeviceInfo, duration_s: int = 10, file_name: str = "video", video_type: VideoType = VideoType.H264):
    save_video(dev, duration_s, file_name, video_type)

def start_simultaenous_recording(duration_s: int = 10, video_type: VideoType = VideoType.H264, file_name: str = "video"):
    devs = ic4.DeviceEnum()
    devices = devs.devices()
    threads = []
    for i, dev in enumerate(devices):
        name = f"{file_name}_{i}"
        print(f"Starting recording for {dev.model_name} ({dev.serial}) to {name}.")
        t = threading.Thread(target=threaded_video_capture, args=(dev, duration_s, name, video_type))
        threads.append(t)
    for t in threads:
        t.start()

    for t in threads:
        t.join()


def get_device(serial: str) -> Optional[ic4.DeviceInfo]:
    """
    Retrieve DevInfo for serial.

    Return None if not found.
    """
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


def main() -> int:
    arguments = argparse.ArgumentParser()

    subs = arguments.add_subparsers(dest="command")
    subs.add_parser("list", help="List devices")

    # args for listing/saving properties

    props = subs.add_parser("properties", help="List properties of device")

    props.add_argument("--json", "-j", help="Print properties serialized", action="store_true")

    props.add_argument("--serial", "-s", help="Serial of the camera", required=True)
    # args for setting properties

    setter = subs.add_parser("set", help="Set properties of device")

    setter.add_argument(
        "--properties", "-p", nargs="+", required=True, help="List of properties to be set. (e.g. -p a=1 b=2)"
    )

    setter.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    # args for saving images

    image = subs.add_parser("image", help="Save image from device")
    image.add_argument("--count", "-c", help="Number of images to save. Default=1", default=1, action="store")
    image.add_argument("--path", help="Directory in which to save images. Default=<current dir>", default=os.getcwd())
    image.add_argument("--name", help="Filename of saved images", default="image-{C}")
    image.add_argument("--type", help="Image types", type=ImageType, choices=list(ImageType), default=ImageType.BMP)

    image.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    # args for saving video

    video = subs.add_parser("video", help="Save video from device")
    video.add_argument(
        "--time", "-t", help="Time period that shall be saved in seconds. Default=10s", default=10, action="store"
    )
    video.add_argument("--name", help="Filename of saved videos", default="video")
    video.add_argument("--type", help="Video types", type=VideoType, choices=list(VideoType), default=VideoType.H264)

    video.add_argument("--serial", "-s", help="Serial of the camera", required=True)
   
    # args for simultaneous recording

    simultaneous = subs.add_parser("sim", help="Start simultaneous recording of multiple devices")
    simultaneous.add_argument(
        "--time", "-t", help="Time period that shall be saved in seconds. Default=10s", default=10, action="store"
    )
    simultaneous.add_argument("--type", help="Video types", type=VideoType, choices=list(VideoType), default=VideoType.H264)
    simultaneous.add_argument("--name", help="Filename of saved videos", default="video")


    # args for live stream

    stream = subs.add_parser("live", help="Display live stream.")

    stream.add_argument("--serial", "-s", help="Serial of the camera", required=True)

    args = arguments.parse_args()

    if len(sys.argv) < 2:
        arguments.print_help()
        return 1
#test
    ic4.Library.init()

    if args.command == "list":
        list_devices()
        return 0

    if args.command != "sim":
        d = get_device(args.serial)
        if not d:
            print("Unable to find device with given serial!")
            return 1

    if args.command == "properties":
        if args.json:
            g = ic4.Grabber()
            g.device_open(d)
            # p = filepath.Path('state.json')
            # g.save_device_state_to_file(p)
            # g.device_save_state_to_file('state.json')
            print(g.device_property_map.serialize().decode("utf-8"))
        else:
            list_properties(d)
    elif args.command == "set":
        set_properties(d, args.properties)
    elif args.command == "image":
        save_image(d, args.count, args.path, args.name, args.type)
    elif args.command == "video":
        save_video(d, args.time, args.name, args.type)
    elif args.command == "live":
        live_stream(d)
    elif args.command == "sim":
        print('Press "q" twice to stop recording.')
        start_simultaenous_recording(args.time, args.type, args.name)
        print("Recording finished.")
    else:
        arguments.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
    # ic4.Library.init()
    # start_simultaenous_recording(2, VideoType.H264)