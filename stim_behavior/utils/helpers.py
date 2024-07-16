from .utils import *
import numpy as np
import pandas as pd
import cv2

def convert_from_seconds_to_frames(interval, fps, time_margin):
    interval[-1][-1] = 0
    interval = np.array(interval)
    interval = interval*fps - time_margin[0]
    interval = interval.tolist()
    interval[-1][-1] = None
    return interval

def find_unit(feature):
    feature = feature.lower()
    if 'angle' in feature:
        return '°'
    elif 'angular speed' in feature:
        return '°/ms'
    elif 'speed' in feature:
        return 'unit'
    else:
        return ''

def fill_nan_linear_interpolation_axis(arr, axis):
    def fill_nan_linear_interpolation(row):
        nan_mask = np.isnan(row)
        nan_len = np.sum(nan_mask)
        indices = np.arange(len(row))
        row[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], row[~nan_mask])
        return row
    return np.apply_along_axis(fill_nan_linear_interpolation, axis, arr)

def compute_L(xy):
    feat = xy
    feat = np.diff(feat, axis=-2)
    feat = np.square(feat)
    feat = np.sqrt(np.sum(feat, axis=-1))
    L = np.sum(feat, axis=-1)
    return L

def find_speed(xy):
    def normalize(speed):
        L = compute_L(xy)
        L = L.mean()
        speed /= L
        return speed

    pose_vel = np.diff(xy, axis=0)
    pose_speed = np.linalg.norm(pose_vel, axis=-1).squeeze()
    pose_speed = np.insert(pose_speed, 0, 0, axis=0)
    pose_speed = normalize(pose_speed)
    return pose_speed

def clean_XY_by_speed(xy):
    pose_speed = find_speed(xy)
    xy[pose_speed > 0.4] = np.nan
    xy = fill_nan_linear_interpolation_axis(xy, axis=0)
    return xy

def find_point_point_distance(xy):
    feat = xy
    feat = np.diff(feat, axis=-2)
    feat = np.square(feat)
    feat = np.sqrt(np.sum(feat, axis=-1))

    L = np.sum(feat, axis=-1, keepdims=True)
    feat /= L

    # IDEAL_LENGTH = np.array([1/2, 1/4, 1/8, 1/8])
    IDEAL_LENGTH = 1/16*np.ones(16)
    feat -= IDEAL_LENGTH
    feat = np.linalg.norm(feat, axis=-1)
    return feat

def clean_XY_by_point_position(xy):
    pp_distance = find_point_point_distance(xy)
    xy[pp_distance > 0.25] = np.nan
    # pdb.set_trace()
    xy = fill_nan_linear_interpolation_axis(xy, axis=0)
    return xy

def load_video(dir, filename):
    video = cv2.VideoCapture(f'{dir}/{filename}.mp4')
    if not video.isOpened():
        # Trying again by appending '-1' to filename
        video = cv2.VideoCapture(f'{dir}/{filename}-1.mp4')
        if not video.isOpened():
            print("File not found:", filename)
            raise FileNotFoundError(filename)
    return video

def load_metadata_new(row, fps=30, time_margin=None):
    if row is None: return np.array([0, None, None])
    stim = int(fps * row['End (s)'])
    arr = np.array([
        stim + time_margin[0],
        stim + time_margin[1],
        stim,
    ])
    return arr

def load_metadata(video_filename, total_frames, row, fps, init_frame=0, time_margin = (-0.5, 6)):
    metadata = {
        'stim': {
            't': None,
            'f': None
        },
        'start': {
            'f': 0
        },
        'end': {
            'f': total_frames
        },
        'title': '',
        'filename': video_filename
    }
    if row is None:
        return metadata

    stim_time = row['End (s)']
    stim_f = int(fps*stim_time)

    start_f = max(0, stim_f - init_frame + int(time_margin[0]*fps))
    end_f = min(total_frames, stim_f - init_frame + int(time_margin[1]*fps))

    classif = row["Classification"]
    stim_type = row["Stimulation Type"]

    metadata = {
        'stim': {
            't': stim_time,
            'f': stim_f
        },
        'start': {
            'f': start_f
        },
        'end': {
            'f': end_f
        },
        'title': f'{classif} - {stim_type}',
        'filename': f'{classif}_{stim_type}_{video_filename}'
    }
    return metadata

def detect_crop_box(pose, frame_shape, threshold=0.9, margin=20):
    detected = pose[:, 2] > threshold

    crop_box = [0, frame_shape[1], 0, frame_shape[0]]

    if np.any(detected):
        x = pose[detected, 0]
        y = pose[detected, 1]

        x1 = int(max([0, int(np.amin(x)) - margin]))
        x2 = int(min([frame_shape[1], int(np.amax(x)) + margin]))
        y1 = int(max([0, int(np.amin(y)) - margin]))
        y2 = int(min([frame_shape[0], int(np.amax(y)) + margin]))

        crop_box = [y1, y2, x1, x2]

    return np.array(crop_box)
    
def detect_pose(dlc, frame, index):
    if index == 0:
        pose = dlc.init_inference(frame)
    else:
        pose = dlc.get_pose(frame)
    return pose

def detect_crop_box_wrapper(dlc, frame, index, threshold=0.9, margin=40):
    pose = detect_pose(dlc, frame, index)
    return detect_crop_box(pose, frame.shape, threshold, margin)

def get_model_path(model_type, dir):
    if model_type == "resnet" :
        return f'{dir}/DLC_Tent6_resnet_50_iteration-0_shuffle-1'
    elif model_type == "mobilenet":
        return f'{dir}/DLC_Tent6_mobilenet_v2_0.35_iteration-0_shuffle-1'
    else:
        return None

def extract_pose_features(pose):
    """ Extracts features from the pose data
    params:
        data: (2, K, 2) np array with prev & curr samples, K keypoints and D=2 (x,y) dimensions
    returns:
        features: (F) F features
    """
    no_of_samples, no_of_points, _ = pose.shape
    pose_vec = np.diff(pose[:, :, :], axis=1)

    feature_mgn = np.linalg.norm(pose_vec, axis=2) # magnitudes of vectors (is it relevant?)

    feature_angles = np.zeros((no_of_samples, no_of_points-2))
    for i in range(no_of_samples):
        for k in range(no_of_points-2):
            feature_angles[i][k] = angle_between(pose_vec[i, k, :], pose_vec[i, k+1, :])

    feature_angles = feature_angles[-1]

    pose_vel = np.diff(pose[:, :, :2], axis=0)
    pose_speed = np.linalg.norm(pose_vel, axis=-1).squeeze()

    features = (feature_angles, pose_speed)
    return features