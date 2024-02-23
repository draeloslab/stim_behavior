import imagingcontrol4 as ic4
import time
def print_device_list():
    print("Enumerating all attached video capture devices...")

    device_list = ic4.DeviceEnum.devices()

    if len(device_list) == 0:
        print("No devices found")
        return

    print(f"Found {len(device_list)} devices:")

    # for device_info in device_list:
    #     print(format_device_info(device_info))

    def format_device_info(device_info: ic4.DeviceInfo) -> str:
        return f"Model: {device_info.model_name} Serial: {device_info.serial}"

def take_picture():
    # Create a Grabber object
    grabber = ic4.Grabber()

    # Open the first available video capture device
    first_device_info = ic4.DeviceEnum.devices()[0]
    grabber.device_open(first_device_info)

    # Set the resolution to 640x480
    grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
    grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)

    # Create a SnapSink. A SnapSink allows grabbing single images (or image sequences) out of a data stream.
    sink = ic4.SnapSink()
    # Setup data stream from the video capture device to the sink and start image acquisition.
    grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

    try:
        # Grab a single image out of the data stream.
        image = sink.snap_single(1000)

        # Print image information.
        print(f"Received an image. ImageType: {image.image_type}")

        # Save the image.
        image.save_as_png("test.png")

    except ic4.IC4Exception as ex:
        print(ex.message)

    # Stop the data stream.
    grabber.stream_stop()


def take_video():
    ic4.Library.init()
    device_list = ic4.DeviceEnum.devices()
    if not device_list:
        print("No devices found.")
        return

    device_info = device_list[0]
    print(f"Using device: {device_info.model_name} (Serial: {device_info.serial})")

    grabber = ic4.Grabber()
    grabber.device_open(device_info.unique_name)

    # Setup video writer
    video_writer = ic4.VideoWriter()
    video_writer_type = ic4.VideoWriterType.MP4_H264
    output_file_name = "recorded_video.mp4"
    video_writer.begin_file(output_file_name, video_writer_type)

    # Create and set up the QueueSink with the listener
    listener = SimpleQueueSinkListener(video_writer)
    sink = ic4.QueueSink(listener)
    grabber.stream_setup(sink)

    grabber.acquisition_start()

    # Wait for a specified duration (e.g., 10 seconds)
    record_duration = 10
    print("Recording...")
    time.sleep(record_duration)

    print("Finishing recording...")
    grabber.acquisition_stop()
    video_writer.finish_file()

    grabber.device_close()
    ic4.Library.exit()
    print("Recording complete.")

class SimpleQueueSinkListener(ic4.QueueSinkListener):
    def __init__(self, video_writer):
        self.video_writer = video_writer

    def sink_connected(self, sink, image_type, min_buffers_required):
        # Allocate the minimum required buffers
        sink.alloc_and_queue_buffers(min_buffers_required)
        return True

    def frames_queued(self, sink):
        # Process available frames
        try:
            while True:  # Keep trying to pop until no more frames are available
                buffer = sink.try_pop_output_buffer()
                if buffer:
                    # Add frame to video file
                    self.video_writer.add_frame(buffer)
                    # Release the buffer back to the queue
                    buffer.release()
                else:
                    break  # Exit loop if no buffer was returned
        except Exception as e:
            print(f"Error processing frame: {e}")

    def sink_disconnected(self, sink):
        print("Sink disconnected.")




if __name__ == "__main__":
    # ic4.Library.init()
    # print_device_list()
    # take_picture()
    take_video()
