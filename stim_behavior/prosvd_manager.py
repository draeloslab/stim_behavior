import numpy as np
import os
from data_manager import DataManager
from proSVD import proSVD
from stim_behavior.utils.utils import rgb_to_grayscale, stream_video, shrink_image
from stim_behavior.utils.logger import Logger


class ProSVDManager():
    """
    A class that provides an interface for working with ProSVD.
    @params
        dim_k (int, required): The number of dimensions to reduce to.
        init_frame (int, optional): The initial number of frames used to initialize prosvd. Defaults to 100.
        target_size (int, optional): The approximate number of pixels per frame to shrink large frames to. This is done to  boost performance. Defaults to 200_000.
        stop_at (int, optional): If specified, only initial `stop_at` framers are processed. Else, all the frames will be processed. Defaults to None.
        log_level (str, optional): Options: ["DEBUG", "INFO", "WARNING", "ERROR"]. Defaults to "DEBUG".
    """

    PROSVD_VIDEO_OUTPUTS = ["dQ", "ld"]

    def __init__(self, dim_k, init_frame=100, target_size=200_000, stop_at=None, log_level="DEBUG"):
        self.dim_k = dim_k
        self.init_frame = init_frame
        self.target_size = target_size
        self.stop_at = stop_at
        self.logger = Logger(log_level=log_level)


    def process_video(self, video_path):
        """
        @params
            video_path (string, required): path to the video directory
        """
        generator = stream_video(video_path, self.logger)

        video_metadata = {"shape": None}
        frames = []  # for proSVD initialization

        dm = DataManager()

        Q_prev = None

        # Iterate over the generator and process each batch
        for index, frame in enumerate(generator):
            frame = rgb_to_grayscale(frame)

            if self.stop_at is not None and index > self.stop_at:
                break

            frame = shrink_image(frame, self.target_size)

            if index == 0:
                video_metadata['shape'] = frame.shape
                self.logger.info(f"Video info [Final size: {frame.shape[1]}*{frame.shape[0]}px]")

            frame = frame.flatten()

            if index < self.init_frame:
                frames.append(frame)
                continue

            if index == self.init_frame:
                frames = np.array(frames).T
                pro = proSVD(k=self.dim_k, w_len=1,history=0, decay_alpha=1, trueSVD=True)
                pro.initialize(frames)

            pro.preupdate()
            pro.updateSVD(frame[:, None])
            pro.postupdate()

            # dm.add('Q', pro.Q)
            Q_curr = pro.Q
            loadings = pro.Q.T@frame
            dm.add('ld', loadings)

            dQ = np.zeros_like(loadings)
            if Q_prev is not None:
                Q_stacked = np.stack([Q_prev, Q_curr])
                Q_diff = np.diff(Q_stacked, axis=0)
                dQ = np.linalg.norm(Q_diff, axis=1)
                dQ = dQ[0] # squeezing: first axis has length 1
            Q_prev = Q_curr

            dm.add('dQ', dQ)

        dm.to_numpy()
        # Q_full = dm.get('Q')
        ld = dm.get('ld')
        dQ = dm.get('dQ')
        del dm
        data = {"dQ": dQ, "ld": ld}

        if data is None or (sorted(data.keys()) != sorted(self.PROSVD_VIDEO_OUTPUTS)):
            self.logger.error("prosvd failed!")
            return
        
        self.logger.debug(f"prosvd gave outputs [{list(data.keys())}]")

        return data, video_metadata
    
    def save_result(self, data, output_dir):
        """
        @params
            data (dict, required): prosvd result
            output_dir (str, required): path to the directory to save prosvd results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for key, val in data.items():
            self.logger.debug(f"Saving [{key}] of shape: [{val.shape}]")
            np.save(f"{output_dir}/{key}.npy", val)
        self.logger.info(f"prosvd outputs saved successfully at {output_dir}")

if __name__ == "__main__":
    video_path = "/home/sachinks/Data/raw/octopus/sample_videos/slight_proximal_touch_220616_122638_000-1.mp4"
    output_dir = "/home/sachinks/Data/tmp"
    dim_k = 4
    stop_at = 150

    prosvdmanager = ProSVDManager(dim_k = dim_k, stop_at = 150)
    data, metadata = prosvdmanager.process_video(video_path = video_path)
    prosvdmanager.save_result(data, output_dir=output_dir)