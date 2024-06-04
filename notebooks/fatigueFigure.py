import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit


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
    data = pd.read_csv("/home/jakejoseph/Desktop/Joseph_Code/FESFatigue-Jake-2024-05-31/videos/testFatigueDLC_resnet50_FESFatigueMay31shuffle1_2000.csv", skiprows=3, header=None)

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

def stimPlot(stimNum,ax, stimArray, index2, color, sliceLength=150, start=20,norm=1):
    stim = stimArray[stimNum]
    subset = index2[stim-start:stim+sliceLength] - index2[stim-start]
    subset = uniform_filter1d(subset, size=4)
    for i, angle in enumerate(subset):
        if angle < 0:
            subset[i] = 0
    subset = subset / norm  #Set to stim 5 max value 
    diff = np.diff(subset)
    stimStart = np.argmax(diff) +8
    stimEnd = np.argmin(diff) -8
    mean = np.mean(subset[stimStart:stimEnd])
    ax.plot(subset, color=color)
    ax.plot( (stimStart+stimEnd)/2, mean, 'ro',label=f"Mean: {mean:.2f}")
    ax.set_title(f"Stim {stimNum}")
    ax.axvline(x=stimStart, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=stimEnd, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Change in Angle (Percent)")
    return mean

def getMean(data,stim, sliceLength=150, start=20):
    subset = data[stim-start:stim+sliceLength] - data[stim-start]
    subset = uniform_filter1d(subset, size=4)
    for i, angle in enumerate(subset):
        if angle < 0:
            subset[i] = 0
    diff = np.diff(subset)
    stimStart = np.argmax(diff) +8
    stimEnd = np.argmin(diff) -8
    mean = np.mean(subset[stimStart:stimEnd])
    return mean

def frame_to_time(frame, fps):
    seconds = frame // fps
    minutes = seconds // 60
    seconds = seconds % 60
    return (f"{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    index2 = np.load('notebooks/NapierFatigueData.npy')
    if 'index2' not in locals():
        data = loadData(threshold=0.5, window=5, span=5, method='exponential')
        index2 = []
        for i in range(len(data)):  #save angle for each posture for all frames
            forearm = (data.iloc[i][data.columns[13]], data.iloc[i][data.columns[14]])
            wrist = (data.iloc[i][data.columns[10]], data.iloc[i][data.columns[11]])
            mcp = (data.iloc[i][data.columns[7]], data.iloc[i][data.columns[8]])
            pip = (data.iloc[i][data.columns[4]], data.iloc[i][data.columns[5]])
            dip = (data.iloc[i][data.columns[1]], data.iloc[i][data.columns[2]])
            index2.append(calculate_angle(wrist, pip, dip))
    stim_array = np.load('notebooks/stim_array.npy')
    on_frames = np.where(np.diff(stim_array.astype(int)) == 1)[0]
    off_frames = np.where(np.diff(stim_array.astype(int)) == -1)[0]


    means = []
    locs =[]
    for on, off in zip(on_frames, off_frames):
        subset = index2[on:off] - index2[on-20]
        means.append(np.median(subset))
        # means.append(np.median(index2[on:off]))
        locs.append( ((off-on)/2) + on)
 
    means = np.array(means)
    means[15:18] = np.nan
    means[-1] = np.nan
    plt.figure(figsize=(20,5))
    plt.plot(index2, label = "W-P-D Angle")
    plt.scatter(locs, means,color= 'r')

    # # Plot vertical lines for on_frames
    # for frame in on_frames:
    #     plt.axvline(x=frame-20, color='r', linestyle='--', alpha=0.5)
    
    # for frame in off_frames:
    #     plt.axvline(x=frame-10, color='b', linestyle='--', alpha=0.5)

    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Angle (Degrees)")
    plt.title("Movement of Index Finger (Interpolated and Exponential Moving Average)")


    # # Create subplots
    # fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)

    # axes = axes.flatten()

    # stimuli = [1, 2, 5, 10, 20, 30, 40, 50]
    # lengths = [150,125,150,160,130,130,120,120]
    # starts = [30, 20, 35, 20, 30, 40, 30, 40]
    # means = []

    # for i, (stim, length, start) in enumerate(zip(stimuli, lengths, starts)):
    #     means.append(getMean(index2, on_frames[stim], length, start))
    # norm = np.max(means)

    # for i, (stim, length, start) in enumerate(zip(stimuli, lengths, starts)):
    #     stimPlot(stim,axes[i], on_frames, index2, 'black', length, start, norm)
    # plt.tight_layout()



    # for i in range(len(on_frames)):
    #     mean = getMean(index2, on_frames[i])
    #     means.append(mean)
    #     print(f"Stimulus {i}: {mean:.2f}")
   
    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    means  = np.array(means)
    mask = ~np.isnan(means)
    filtered_means = means[mask]
    filtered_indices = np.arange(len(means))[mask]
    # Perform curve fitting

    popt, pcov = curve_fit(exponential_decay, filtered_indices, filtered_means, maxfev=10000)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(means)), means,label='Original Data', color = 'r',)
    plt.plot(filtered_indices, exponential_decay(filtered_indices, *popt), label=f'{popt[0]:.2f} * exp^(-{popt[1]:.2f} * x) + {popt[2]:.2f}')
    plt.xlabel("Stimulus Number")
    plt.ylabel("Mean Change in Angle (Percent)")
    plt.legend()
    plt.show()

