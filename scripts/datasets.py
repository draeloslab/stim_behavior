# TODO: move the logic in musall-mice/data-preprocessing/download_data.py to this file after discussing Jonathan's insights (AdaptiveLatent)

class Musall19Dataset():
    doi = 'https://doi.org/10.1038/s41593-019-0502-4'
    automatically_downloadable = False

    def __init__(self, cam=1, video_target_dim=100, resize_factor=1):
        self.cam = cam # either 1 or 2
        self.video_target_dim = video_target_dim
        self.resize_factor = resize_factor