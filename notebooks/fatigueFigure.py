import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def movingAverage(data, window, method='linear'):
    for i in range(1, len(data.columns), 3):
        x = data.columns[i]
        y = data.columns[i+1]
        
        if method == 'linear':
            data[x] = data[x].rolling(window=window, min_periods=1).mean()
            data[y] = data[y].rolling(window=window, min_periods=1).mean()
        elif method == 'exponential':
            data[x] = data[x].ewm(span=window, adjust=False).mean()
            data[y] = data[y].ewm(span=window, adjust=False).mean()
        else:
            raise ValueError("Method must be 'linear' or 'exponential'")

    return data

def interpolate_columns(df, threshold=0.90):
    for i in range(1, len(df.columns), 3):
        likelihood_col = i + 2
        x_col = i
        y_col = i + 1
        mask = df[likelihood_col] < threshold
        df.loc[mask, [df.columns[x_col], df.columns[y_col]]] = df.loc[~mask, [df.columns[x_col], df.columns[y_col]]].interpolate(method='linear')
    return df

def calculate_angle(p1,p2, p3):
    #Calculate the angle at p2
    #Calculate the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    #Calculate the angle
    angle = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    #Convert to degrees
    angle = np.degrees(angle)

    return angle

def interpolatePoint(p1, p2, t):
    x1, y1 = p1
    x2, y2 = p2
    
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    
    return (x, y)

def interpolate(df, x, y, likelihood, threshold):
    for i in range(1, len(df)):
        if df.iloc[i, likelihood] < threshold:
            df.iloc[i, x] = df.iloc[i-1, x]
            df.iloc[i, y] = df.iloc[i-1, y]

def loadData(threshold=0.5, window=5, span=5, method='linear'):
    #Load extracted x,y data
    data = pd.read_csv("/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/videos/MVI_0401DLC_resnet50_FES_V1Oct16shuffle1_38000.csv", skiprows=3, header=None)

    for i in range(1,len(data.columns),3):
        likelihood = i + 2  
        x_col = i  
        y_col = i + 1  
        interpolate(data, x_col, y_col, likelihood, threshold)

        #moving average
    for i in range(1, len(data.columns),3):
        x = data.columns[i]
        y = data.columns[i+1]
        if method == 'linear':
            data[x] = data[x].rolling(window=window, min_periods=1).mean()
            data[y] = data[y].rolling(window=window, min_periods=1).mean()
        elif method == 'exponential':
            x_ma = data[x].ewm(span=span, adjust=False).mean()
            y_ma = data[y].ewm(span=span, adjust=False).mean()
            data[x] = x_ma
            data[y] = y_ma
    return data

def stimPlot(ax, stim, index2, color, sliceLength=150, start=20):
    subset = index2[stim-start:stim+sliceLength] - index2[stim-start]
    subset = uniform_filter1d(subset, size=4)
    for i, angle in enumerate(subset):
        if angle < 0:
            subset[i] = 0
    subset = subset / 83.92148926983948  #Set to stim 5 max value 
    ax.plot(subset, color=color)
    ax.set_title(f"Stim at {frame_to_time(stim, 30)}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Change in Angle (Percent)")
    return subset

def frame_to_time(frame, fps):
    seconds = frame // fps
    minutes = seconds // 60
    seconds = seconds % 60
    return (f"{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    stim_array = np.load('notebooks/stim_array.npy')
    index2 = np.load('notebooks/napierFatigueData.npy')
    on_frames = np.where(np.diff(stim_array.astype(int)) == 1)[0]

    plt.figure(figsize=(20,5))
    plt.plot(index2, label = "W-P-D Angle")

    # Plot vertical lines for on_frames
    for frame in on_frames:
        plt.axvline(x=frame, color='r', linestyle='--', alpha=0.5)

    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Angle (Degrees)")
    plt.title("Movement of Index Finger (Interpolated and Exponential Moving Average)")
    plt.show()

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)

    axes = axes.flatten()

    stimuli = [1, 2, 5, 10, 20, 30, 40, 50]
    lengths = [150,150,150,160,130,130,120,120]
    starts = [30, 20, 35, 20, 30, 40, 30, 40]

    for i, (stim, length, start) in enumerate(zip(stimuli, lengths, starts)):
        stimPlot(axes[i], on_frames[stim], index2, 'black', length, start)

    plt.tight_layout()
    plt.show()
