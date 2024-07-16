from dlclive import DLCLive, Processor
from stim_behavior.data_manager import DataManager


class DLCManager:
    def __init__(self, 
            model_path,
            processor=Processor(),
            pcutoff=0.2,
            resize=1):
        
        self.model = DLCLive(
            model_path=model_path,
            processor=processor,
            pcutoff=pcutoff,
            # display=True,
            resize=resize)
        self.frame = None
        self.is_first_frame = None
        self.dm = DataManager()

    def init_data(self, feature_keys):
        self.dm = DataManager(feature_keys)
        self.prev_pose_xy = None
        
    def update_frame(self, frame, is_first_frame):
        self.frame = frame
        self.is_first_frame = is_first_frame
        
    def detect_pose_helper(self):
        if self.is_first_frame:
            pose = self.model.init_inference(self.frame)
        else:
            pose = self.model.get_pose(self.frame)
        return pose
        
    def detect_pose(self):
        curr_pose = self.detect_pose_helper()
        
        curr_pose_xy, curr_pose_p = curr_pose[:, :-1], curr_pose[:, -1]
        if self.prev_pose_xy is None:
            self.prev_pose_xy = curr_pose_xy

        pose = np.stack([self.prev_pose_xy, curr_pose_xy])
        feature_angles_item, pose_speed_item = extract_pose_features(pose)

        self.dm.add('xy', curr_pose_xy)
        self.dm.add('p', curr_pose_p)
        self.dm.add('angles', feature_angles_item)
        self.dm.add('speed', pose_speed_item)

        self.prev_pose_xy = curr_pose_xy

    def save_data(self, dir):
        self.dm.to_numpy()
        self.dm.save(dir)

    def load_data(self, dir, feature_keys):
        return self.dm.load(dir, feature_keys)