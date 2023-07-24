from utils import *

def get_stim_location(stim_type):
    stim_type = stim_type.lower()
    location = "Unknown"
    locations = ["Distal" ,"Proximal", "Cord"]
    for loc in locations:
        if loc.lower() in stim_type:
            location = loc
            break
    return location

def get_stim_method(stim_type):
    stim_type = stim_type.lower()
    stimulus = "Unknown"
    if ("touch" in stim_type) or ("pinch" in stim_type):
        stimulus = "Mechanical"
    elif ("electrical" in stim_type) or ("hz" in stim_type) or ("one stimulation" in stim_type):
        stimulus = "Electrical"
    return stimulus

def load_metadata_new(row, fps=30, time_margin = (-15, 180)):
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

def read_octopus_xlsx(xls_path):
    df = pd.read_excel(xls_path) 
    df2 = df.copy()
    df = df.drop(index=0)
    df = df.rename(columns=df2.iloc[0])
    df['Stim Location'] = df['Stimulation Type'].apply(get_stim_location)
    df['Stim Method'] = df['Stimulation Type'].apply(get_stim_method)
    df['Stimulation Class'] = df['Stim Location'] + ' ' + df['Stim Method']
    df.reset_index(inplace=True)
    return df

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