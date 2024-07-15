# Video is divided into 16 pieces (4x4) hence we need to restore each of them and piece them together

import numpy as np
import os
from scipy.io import loadmat
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import cv2

class MiceDataLoader:
    def __init__(self, raw_dir, processed_dir, verbose=True):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.verbose = verbose
        if self.verbose:
            attributes = vars(self)
            for key, value in attributes.items():
                print(f"{key}: {value}")

    def update_verbose(self, verbose):
        self.verbose = verbose

    def init_file(self, mouse_id, date, cam):
        self.mouse_id = mouse_id
        self.date = date
        self.cam = cam
        self.processed_filename = f'{mouse_id}_{date}_cam{cam}'
        if self.verbose:
            attributes = vars(self)
            for key, value in attributes.items():
                print(f"{key}: {value}")
    
    def merge_svd(self, start_V=0, end_V=1000, rf = 1, save_data=False):
        '''
        rf: resize factor
        '''
        file_dir = f'{self.raw_dir}/2pData/Animals/{self.mouse_id}/SpatialDisc/{self.date}/BehaviorVideo'

        used_V = None
        if (start_V is not None) and (end_V is not None):
            used_V = end_V - start_V

        Wid, Hei = 320, 240
        Wid0, Hei0 = Wid//4, Hei//4

        data = [[] for _ in range(4)]

        for k in tqdm(range(16)):
            name = f'{file_dir}/SVD_Cam{self.cam}-Seg{k+1}.mat'
            # Load MATLAB .mat file
            mat_contents = loadmat(name)
            V = mat_contents['V'] # (89928, 500)
            U = mat_contents['U'] # (500, 4800)

            if used_V is None: used_V = V.shape[0]
            
            VU = V[start_V:end_V, :].dot(U) # (T, 4800)
            seg = VU.reshape((used_V, Wid0, Hei0))
            Wid1, Hei1 = Wid0//rf, Hei0//rf
            seg = resize(seg, (used_V, Wid1, Hei1), mode='constant')
            data[k//4].append(seg)


        # for k in range(16):
        #     i, j = k//4, (k%4)
        #     # Data[:, i * Wid1: (i + 1) * Wid1, j*Hei1 : (j + 1) * Hei1] = seg

        for i in range(4):
            data[i] = np.concatenate(data[i], axis=-1)
        data = np.concatenate(data, axis=-2)

        if self.verbose:
            plt.imshow(data[0], cmap='gray')

        self.h5_saved = save_data
        if save_data:
            ## save HDF5 file
            output_path = f'{self.processed_dir}/h5/{self.processed_filename}.h5'

            # Check if the directory exists, and if not, create it
            directory = os.path.dirname(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            with h5py.File(output_path, 'w') as hf:
                hf.create_dataset('data', data=data)

            self.h5_path = output_path
            print('HDF5 saved successfully at', self.h5_path)
        else:
            self.h5_file = data

    def get_filename(self):
        return self.processed_filename

    def get_videofilepath(self):
        return f'{self.processed_dir}/video/{self.processed_filename}.mp4'

    def save_video(self, fps = 15):
        output_path = self.get_videofilepath()

        # Check if the directory exists, and if not, create it
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.h5_saved:
            with h5py.File(self.h5_path, 'r') as hf:
                data = hf['data'][:]
        else:
            data = self.h5_file

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

        # Get image dimensions from the data
        height, width = data.shape[1], data.shape[2]

        # Create a VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Convert data to uint8 and write each frame to the video
        for frame in data:
            frame = (frame).astype(np.uint8)  # Convert to uint8 for display
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
            out.write(frame)

        # Release the VideoWriter and display a message
        out.release()
        print('Video saved successfully at', output_path)

    def save_gif(self):
        from PIL import Image

        output_path = f'{self.processed_dir}/gif/{self.processed_filename}.gif'
        
        # Check if the directory exists, and if not, create it
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.h5_saved:
            with h5py.File(self.h5_path, 'r') as hf:
                data = hf['data'][:]
        else:
            data = self.h5_file

        # Convert data to uint8 (assuming grayscale images)
        data = (data).astype('uint8')

        # Prepare images for GIF creation
        images_for_gif = []
        for frame in data:
            img = Image.fromarray(frame)
            images_for_gif.append(img)

        # Save the images as a GIF
        images_for_gif[0].save(
            output_path,
            save_all=True,
            append_images=images_for_gif[1:],
            loop=0,
            duration=100  # Duration is in milliseconds
        )

        print('GIF saved successfully at', output_path)

        