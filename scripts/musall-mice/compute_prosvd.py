import argparse
import sys
import h5py
from proSVD import proSVD
import numpy as np
import matplotlib.pyplot as plt

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--project_root', type=str, required=True,
      help='Root directory of the project. (eg: "/home/<username>/Code/stim_behavior/")')
    parser.add_argument( '--input_dir', type=str, required=True,
      help='Directory where the videos are stored. (eg: "/home/<username>/Data/processed/mouse-cshl")')
    parser.add_argument( '--output_dir', type=str, required=True,
      help='Directory to store the prosvd data. (eg: "/home/<username>/Data/output/mouse-cshl")')
    return parser


##################################
def compute_prosvd():
    generator = stream_video(data_ld.get_videofilepath())

    frames = []  # for proSVD initialization

    # Iterate over the generator and process each batch
    for index, frame in enumerate(generator):
        frame = rgb_to_grayscale(frame)

        if index > 500:
            break

        if index == 0:
            video_metadata['shape'] = frame.shape

        frame = frame.flatten()

        if index < PROSVD_INIT_FRAME:
            frames.append(frame)
            continue

        if index == PROSVD_INIT_FRAME:
            frames = np.array(frames).T
            pro = proSVD(k=PROSVD_K, w_len=1,history=0, decay_alpha=1, trueSVD=True)
            pro.initialize(frames)

        pro.preupdate()
        pro.updateSVD(frame[:, None])
        pro.postupdate()

        dm.add('Q', pro.Q)
        dm.add('S', pro.S)
        ld = pro.Q.T@frame # loadings
        dm.add('ld', ld)
    dm.to_numpy()
##################################

args = create_parser().parse_args()
sys.path.append(args.project_root)
from utils.utils import *
from packages.data_manager import DataManager
from models.mice_data_loader import MiceDataLoader

processed_dir = '/home/sachinks/Data/processed/mouse-cshl'

data_ld = MiceDataLoader(None, processed_dir)

###
mouse_id = 'mSM49'
date = '14-Sep-2018'
cam = '2'

data_ld.init_file(mouse_id, date, cam)


video = cv2.VideoCapture(data_ld.get_videofilepath())
fps = video.get(cv2.CAP_PROP_FPS)


PROSVD_K = 4 # no. of dims to reduce to
PROSVD_INIT_FRAME = 100 # No of initial frames used to initialize proSVD

video_metadata = {
    'shape': None,
    'fps': fps
}

dm = DataManager()
compute_prosvd()


# Saving proSVD outputs

output_dir = '/home/sachinks/Data/output/mouse-cshl'
output_dir = f'{output_dir}/{data_ld.get_filename()}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dm.save(output_dir)


# Visualization of proSVD

Q = dm.get('Q')
frame_shape = video_metadata['shape']
i = 0
Qi = Q[i]
Qi = Qi.reshape(frame_shape[0], frame_shape[1], PROSVD_K)
plt.figure(figsize=(3, 3))
plt.imshow(Qi[..., 0], cmap='gray')
plt.show()