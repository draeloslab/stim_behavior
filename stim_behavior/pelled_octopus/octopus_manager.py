import numpy as np
import yaml
import json
import os

from stim_behavior.deeplabcut_manager import DLCManager
from stim_behavior.utils.logger import Logger
# from stim_behavior.utils.utils import *
from .utils import read_octopus_xlsx

class Driver:
    def __init__(self, config_file, dev_mode = False, video_type=None, verbose = True, log_level="DEBUG"):
        self.logger = Logger(log_level=log_level)
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        #####################
        # new code
        video_dir = "/nfs/turbo/umms-adraelos/sachinks/stim_behavior/sample_videos"
        metadata_path = "/nfs/turbo/umms-adraelos/sachinks/stim_behavior/metadata.json"
        model_path = "/nfs/turbo/umms-adraelos/sachinks/stim_behavior/models/octopus_Octo-Arm_resnet_50_keypoint17"

        with open(metadata_path, 'r') as json_file:
            metadata_json = json.load(json_file)
        self.fps = metadata_json['fps']
        self.video_dir = video_dir
        self.set_filenames(metadata_path=config['path']['xls'])


        # self.working_dir = f"{root_dir}/{config['path']['working']}" # to save processed data and figures

        self.feature_keys = ['xy', 'p', 'angles', 'speed']
        self.dlc = DLCManager(model_path)
        
        self.prosvd_k = 4 # no. of dims to reduce to

        self.time_margin = (-20, 180) # to trim videos
        self.total_f = self.time_margin[1] - self.time_margin[0]

        self.tx = np.arange(self.time_margin[0], self.time_margin[1])/self.fps

        self.stim_class_list =  ['Cord Electrical', 'Distal Electrical', 'Proximal Electrical', 'Cord Mechanical', 'Distal Mechanical', 'Proximal Mechanical']

        self.movement_types = [
            "No movement",
            "Movement",
            "Movement with arm curl"
        ]
    
    def filter_data(self, stim_method = None):
        """@params
            stim_method: options -> ["Mech", "Elec"]
        """
        if not self.is_metadata_present:
            return
        old_count = len(self.filenames)
        if stim_method == "Mech":
            self.files_info = self.files_info[self.files_info['Stim Method'] == 'Mechanical']
        elif stim_method == "Elec":
            self.files_info = self.files_info[self.files_info['Stim Method'] == 'Electrical']
        else:
            pass
        self.filenames = self.files_info["File Name"].to_list()
        new_count = len(self.filenames)
        if new_count == old_count:
            self.logger.warning(f"No filtering done")
        elif new_count == 0:
            self.logger.warning(f"New count is 0. No files will be processed.")
        else:
            self.logger.info(f"Filenames reduced from {new_count} to {old_count}")

    def set_filenames(self, metadata_path=None):
        filenames_from_videodir = [os.path.splitext(file)[0] for file in os.listdir(self.video_dir)]
        self.is_metadata_present = (metadata_path is not None)
        if self.is_metadata_present:
            xls_path = metadata_path
            self.files_info = read_octopus_xlsx(xls_path)   
            filenames_from_metadata = self.files_info["File Name"].to_list()       
            self.filenames = list(set(filenames_from_metadata) & set(filenames_from_videodir))
            self.files_info = self.files_info[self.files_info['File Name'].isin(self.filenames)]
        else:
            self.filenames = filenames_from_videodir

        self.logger.info(f"Processing {len(self.filenames)} videos from {self.video_dir}")
        if len(self.filenames) < 4:
            # self.logger.debug('\t', end='')
            self.logger.debug(*self.filenames, sep="\n\t")

    def get_fig_dir(self, filename):
        figs_dir = f"{self.working_dir}/figs"
        os.makedirs(figs_dir, exist_ok=True)
        return figs_dir

    def get_data_dir(self, filename=None):
        if filename is None:
            data_dir = f"{self.working_dir}"
        else:
            data_dir = f"{self.working_dir}/data/{filename}"
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    def load_video(self, video_idx):
        video_filename = self.filenames[video_idx]

        self.video = None
        try:
            self.video = load_video(self.video_dir, video_filename)
        except:
            return False

        self.index = -1

        self.dlc.init_data(feature_keys=self.feature_keys)

        return (self.video is not None)
    
    def is_video_empty(self):
        return self.video is None or not self.video.isOpened()
    
    def read_video(self):
        ret, self.frame = self.video.read()
        if not ret: return False
        # if not self.frame:
        #     pdb.set_trace()
        self.frame = self.frame[:,:,::-1]
        self.index += 1
        self.dlc.update_frame(self.frame, self.index == 0)
        return ret
    
    def release_video(self):
        self.video.release()
    
    def detect_pose(self):
        self.dlc.detect_pose()

    def save_data(self, video_idx):
        video_filename = self.filenames[video_idx]
        data_dir = self.get_data_dir(video_filename)
        self.dlc.save_data(data_dir)

    def post_process(self, verbose=False):
        err_log = {
            'poor_pose': [],
            'file_missing': []
        }

        columns = ['filename', *self.feature_keys, 'move_class', 'stim_class']
        df = pd.DataFrame(columns=columns)

        for video_idx in range(len(self.filenames)):
            video_filename = self.filenames[video_idx]

            data_dir = self.get_data_dir(video_filename)

            try:
                features = self.dlc.load_data(data_dir, self.feature_keys)
            except ValueError:
                err_log['poor_pose'].append(video_filename)
                continue
            except FileNotFoundError:
                err_log['file_missing'].append(video_filename)
                continue
            except:
                raise("Uncaught exception")
            
            row = None
            if self.is_metadata_present:
                row = self.files_info.iloc[video_idx]
            md = load_metadata_new(row, time_margin = self.time_margin)

            start_f, end_f = md[0], md[1]

            data = {}

            for key, value in features.items():
                data[key] = value[start_f: end_f, ...]

            move_idx, stim_idx = 0, 0
            if self.is_metadata_present:
                move_idx = int(row['Classification'])
                stim_class = row["Stimulation Class"]
                stim_idx =  self.stim_class_list.index(stim_class)

            data['filename'] = video_filename
            data['move_class'] = move_idx
            data['stim_class'] = stim_idx

            df = df.append(data, ignore_index=True)

        if verbose:
            for key, items in err_log.items():
                print(f"{key}: {len(items)}")
                for item in items:
                    print(f"\t{item}")

        data_dir = self.get_data_dir()

        feature_key_tuple = [('features', key) for key in self.feature_keys]

        df.columns = pd.MultiIndex.from_tuples([
            ('metadata', 'filename'),
            *feature_key_tuple,
            ('labels', 'move_class'),
            ('labels', 'stim_class')
        ])

        with open(f'{data_dir}/dlc_features.pkl', 'wb') as f:
            pickle.dump(df, f)

    def visualize_results(self):
        num_lines = 3 # for 3 angles
        colors = plt.cm.Paired(np.linspace(0, 1, num=num_lines))

        num_rows, num_cols = 1, 3
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 4), gridspec_kw={'top': 0.85})

        for i, key in enumerate(self.feature_angles_dict.keys()):
            feature_angle_mean = np.mean(self.feature_angles_dict[key], axis=0)

            for k in range(3):
                data = feature_angle_mean[..., k]
                data = smooth_data(data, 5)
                axs[i].plot(data, label=k, linewidth=2, c=colors[k])
            axs[i].set_ylim(-10, 190)
            axs[i].set_title(key)
            axs[i].set_xlabel("Time (s)")
            axs[i].axvline(x=0, color='orange', linewidth=2, alpha=0.3)

        legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=i) for i in range(3)]
        fig.legend(title="Angle", handles=legend_elements, loc='upper right')
        fig.suptitle(f'Electrical Stimulations - Angle')

        figs_dir = self.get_fig_dir("")
        figs_dir_full = f'{figs_dir}/dlc-summary'
        os.makedirs(figs_dir_full, exist_ok=True)
        fig.savefig(f'{figs_dir_full}/Electrical Stimulations - Angle.png', facecolor='white')