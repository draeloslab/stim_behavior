import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter1d
from skimage import transform
import h5py
from scipy import ndimage

def downsample_image(img, sz=200):
    # Define the desired output size
    output_size = (sz, sz)

    # Calculate the scaling factors
    height_scale = output_size[0] / img.shape[0]
    width_scale = output_size[1] / img.shape[1]

    # Determine the scaling factor to preserve aspect ratio
    scale_factor = min(height_scale, width_scale)

    # Calculate the new dimensions based on the scale factor
    new_height = int(img.shape[0] * scale_factor)
    new_width = int(img.shape[1] * scale_factor)

    # Perform downsampling while preserving aspect ratio
    downsampled_img = transform.resize(img, (new_height, new_width), preserve_range=True).astype(np.uint8)

    return downsampled_img

def load_video(dir, filename, buffer_size = 10):
    video = cv2.VideoCapture(f'{dir}/{filename}.mp4')
    if not video.isOpened():
        # Trying again by appending '-1' to filename
        video = cv2.VideoCapture(f'{dir}/{filename}-1.mp4')
        if not video.isOpened():
            print("File not found:", filename)
            raise FileNotFoundError(filename)
    
    # Set your desired buffer size
    video.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return video

def reshape_to_2d(frame_flat, box):
    w = box[1]-box[0]
    h = box[3]-box[2]
    frame_rect = frame_flat.reshape(w, h)
    return frame_rect

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def smooth_data(data, kernel_size = 20):
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = ndimage.convolve(data, kernel)
    return data_convolved

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

def add_time_margin(orig_start_end_time, total_time, time_margin = 1):
    """
    Adds time margin before and after movement

    @params
    total_time: in seconds. Total time of the video
    time_margin: in seconds. Time margin to keep before and after movement
    """
    start_end_time = np.copy(orig_start_end_time)
    start_end_time[0] -= time_margin
    start_end_time[1] += time_margin

    start_end_time[0] = max(start_end_time[0], 0)
    start_end_time[1] = min(start_end_time[1], total_time)

    return start_end_time

def rgb_to_grayscale(image):
    # Compute the grayscale image using the luminosity formula
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    grayscale_image = grayscale_image.astype(np.uint8)

    return grayscale_image

def load_video_as_npy(video_path, working_dir, video_filename):
    numpy_path = f'{working_dir}/{video_filename}.npy'
    convert_mp4_to_npy(video_path, numpy_path)

    h5_path = f'{working_dir}/{video_filename}.h5'
    convert_npy_to_h5(numpy_path, h5_path, 'data')

    f = h5py.File(h5_path, 'r')
    video_h5 = np.array(f['data'])
    f.close()

    return video_h5

def convert_poses_h5_to_npy(df):
    d = 3 # x, y, likelihood
    k = df.shape[1]//d # no of keypoints
    n = df.shape[0] # no of samples/frames
    return df.to_numpy().reshape(n, k, d)

def read_octopus_xlsx(xlxs_path):
    df = pd.read_excel(xlxs_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])
    return df

def convert_xlsx_to_csv(xlxs_path, csv_path):
    df = pd.read_excel(xlxs_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])
    df.to_csv(csv_path, index=False)
    metadata = {
        'fps': int(df2.keys()[1])
    }
    return metadata

def convert_mp4_to_npy_as_grayscale(video_path, output_path=None):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an empty numpy array to store the frames
    frames = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))

    # Read and save each frame
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        frame = rgb_to_grayscale(frame)
        frames[i] = frame

    # Save the frames as NPY file
    if output_path is not None:
        np.save(output_path, frames)

    # Release the video capture object
    video.release()

    return frames

def convert_mp4_to_npy(video_path, output_path=None):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an empty numpy array to store the frames
    frames = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    # Read and save each frame
    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        frames[i] = frame

    # Save the frames as NPY file
    if output_path is not None:
        np.save(output_path, frames)

    # Release the video capture object
    video.release()

    return frames

def convert_npy_to_h5(npy_file, h5_file, dataset_name):
    # Load the numpy array
    array = np.load(npy_file)

    # Create a new HDF5 file
    with h5py.File(h5_file, 'w') as f:
        # Create a dataset and write the array to it
        f.create_dataset(dataset_name, data=array)

def gaussian_filter_time(data):
    smt = np.zeros(data.shape)
    for filti in range(data.shape[0]):
        smt[filti, :] = gaussian_filter1d(data[filti, :].astype('float'), sigma=2)
    return smt