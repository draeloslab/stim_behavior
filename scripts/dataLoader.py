from matplotlib.animation import FFMpegWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import cv2
from utils import calculate_angle

class DataLoader:
        def __init__(self, name, file, threshold, window,type='DLC'):
            self.file = file
            self.name = name
            self.threshold = threshold
            self.window = window
            self.type = type
            self.wrist, self.index = self.loadData()

        def loadData(self):

            data = pd.read_csv(self.file, skiprows=3, header=None)

            def interpolate(df, x, y, likelihood, threshold):
                for i in range(1, len(df)):
                    if df.iloc[i, likelihood] < threshold:
                        df.iloc[i, x] = df.iloc[i-1, x]
                        df.iloc[i, y] = df.iloc[i-1, y]

            # Linear Interpolation based on likelihood threshold

            if self.type == 'DLC':
                step =1
            else:
                step = 3  # NOTE: This is for sleap since the first 3 columns are not coordinates

            for i in range(step,len(data.columns),3):
                likelihood = i + 2  
                x_col = i  
                y_col = i + 1  
                interpolate(data, x_col, y_col, likelihood, self.threshold)

            # Moving average
            for i in range(step, len(data.columns),3):
                x = data.columns[i]
                y = data.columns[i+1]
                data[x] = data[x].rolling(window=self.window, min_periods=1).mean()
                data[y] = data[y].rolling(window=self.window, min_periods=1).mean()        

            wristAngle = []
            indexAngle = []
            self.x = []
            self.y = []
            for i in range(len(data)):  #save angle for each posture for all frames
                if self.type == 'DLC':
                    forearm = (data.iloc[i][data.columns[13]], data.iloc[i][data.columns[14]])
                    wrist = (data.iloc[i][data.columns[10]], data.iloc[i][data.columns[11]])
                    mcp = (data.iloc[i][data.columns[7]], data.iloc[i][data.columns[8]])
                else:
                    forearm = (data.iloc[i][data.columns[15]], data.iloc[i][data.columns[16]])
                    wrist = (data.iloc[i][data.columns[12]], data.iloc[i][data.columns[13]])
                    mcp = (data.iloc[i][data.columns[9]], data.iloc[i][data.columns[10]])
                    pip = (data.iloc[i][data.columns[6]], data.iloc[i][data.columns[7]])
                    dip = (data.iloc[i][data.columns[3]], data.iloc[i][data.columns[4]])
                # Unpack the points into x and y coordinates
                self.x.append([forearm[0], wrist[0], mcp[0], pip[0], dip[0]])
                self.y.append([forearm[1], wrist[1], mcp[1], pip[1], dip[1]])
                wristAngle.append(calculate_angle(forearm, wrist, mcp))
                indexAngle.append(calculate_angle(wrist, mcp, pip)) # NOTE: This is using PIP as a vertex, some might use DIP
            return np.array(wristAngle), np.array(indexAngle)


class FatigueAnalysis(DataLoader):
    def __init__(self, name, file, threshold, window, type='DLC',peakWindow=20,height=130, prominence=10, width=10, distance=10):
        super().__init__(name, file, threshold, window, type)
        self.data = self.wrist  #Only using wrist angle for this paper
        self.peakWindow = peakWindow
        self.height = height
        self.prominence = prominence
        self.width = width
        self.distance = distance
        self.getStimAngle()

    def getStimAngle(self):
        self.peaks, _ = find_peaks(self.data, height=self.height, prominence=self.prominence, width=self.width, distance=self.distance)  

        angleChanges = []
        for i in range(len(self.peaks)):
            max = np.max(self.data[self.peaks[i]-self.peakWindow:self.peaks[i]+self.peakWindow])
            angleChanges.append(max) # - self.data[self.valleys[-restIdx]])
        self.angleChanges = angleChanges

        # Plot to visually inspect if peaks were correctly detected
        plt.figure(figsize=(25, 5))
        plt.plot(self.data)
        plt.plot(self.peaks, self.angleChanges, ".r", label='Peaks')
        plt.title(self.name)
        plt.xlabel("Frames")
        plt.ylabel("Absolute Angle (Degrees)")
        plt.show()

    def stimProfiles(self, stimuli, lengths, starts):

        fig, axes = plt.subplots(1, length(stimuli), figsize=(20, 5), sharex=True, sharey=True)
        axes = axes.flatten()

        def plotSingleStim(stimNum,ax, length, start):
            stim = self.peaks[stimNum]
            subset = self.data[stim-start:stim+length]
            ax.plot(subset, color='black')
            ax.set_title(f"Stim {stimNum+1}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Absolute Angle (degrees)")


        for i, (stim, length, start) in enumerate(zip(stimuli, lengths, starts)):
            plotSingleStim(stim, axes[i], length,start)
            axes[i].set_ylim(110, 200)  
            axes[i].axhline(y=180, color='red', linestyle='--')
            plt.suptitle('Napier')
            plt.ylabel('Absolute Angle (degrees)')
            plt.tight_layout()
            plt.show()
  

    def fitExponentialDecay(self, guess, removeStartSlice = 0):

        # Fit exponential decay to means
        def exponential_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        def double_exponential_decay(x, a, b, c, d, e):
            return a*np.exp(-b*x) + c*np.exp(-d*x) + e

        # bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

        # popt1, pcov = curve_fit(double_exponential_decay, self.filtered_indices[removeStartSlice:], self.filtered_means[removeStartSlice:], maxfev=10000,p0=[1,0,5,0,160], bounds=bounds)
        # self.double_exponential = double_exponential_decay(self.filtered_indices, *popt1)

        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        popt, _ = curve_fit(exponential_decay, self.filtered_indices[removeStartSlice:], self.filtered_means[removeStartSlice:], maxfev=10000, p0=guess, bounds=bounds)
        self.exponential = exponential_decay(self.filtered_indices[removeStartSlice:], *popt)
        return popt

    def makeVideo(self, video_file, saveName,stimNum, start, sliceLength):

        stim = self.peaks[stimNum]
        startFrame = stim - start
        endFrame = stim + sliceLength

        # clipLength = n *30

        # Open the video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error opening video file")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

        # Set up the writer object to write your video
        writer = FFMpegWriter(fps=30)

        # Create a figure for plotting
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        fig.patch.set_facecolor('black')  # Set background color to black
        plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0)  # Adjust subplots to minimize borders

        # Prepare the video file to write to
        with writer.saving(fig, saveName, 100):
            i = startFrame

            while cap.isOpened() and i < endFrame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
                # Small white dots with thick lines
                ax.plot(self.x[i], self.y[i], 'o-', color='white', markersize=3, linewidth=5)
                # Large white markers with thick black edges
                ax.scatter(self.x[i], self.y[i], c='white', s=300, linewidth=3, edgecolors='black',zorder=10)

   
        # clipLength = n *30
             

                # Fixed position for the angle text (in axis coordinates)
                x_pos = 0.7  # 80% from the left of the figure
                y_pos = 0.7  # 50% from the bottom of the figure

                # Create a smaller rectangle patch
                rect = patches.Rectangle((x_pos-0.08, y_pos-0.05), 0.16, 0.1,
                                        facecolor='gray', alpha=0.5,
                                        transform=ax.transAxes)

                # Add the rectangle to the plot
                ax.add_patch(rect)

                # Add the text on top of the rectangle
                ax.text(x_pos, y_pos,
                        f'Angle: {self.data[i]:.0f}',
                        color='white', fontsize=12,
                        ha='center', va='center',
                        transform=ax.transAxes)

                fig.suptitle(f'Monkey {self.name[:1]} Stim {stimNum+1}', 
                            ha='center', va='top', fontsize=20, weight='bold', color='white')
                
                # ax.text(0.5, 0.95, f'Max Angle: {self.angleChanges[stimNum]:.0f}', color='white', fontsize=16, ha='center', va='top', transform=ax.transAxes)
                ax.axis('off')
                # plt.tight_layout()
                # Write the current frame to the video
                writer.grab_frame()
                ax.clear()

                i += 1

        # Release the video capture object and close the figure
        cap.release()
        plt.close(fig)

if __name__ == "__main__":
    # Create an instance of FatigueAnalysis
    fatigue_analysis = FatigueAnalysis("Monkey", "data.csv", 0.5, 5)
    
    # Call the necessary methods
    fatigue_analysis.getAngleChange(10)
    fatigue_analysis.plotAll()
    fatigue_analysis.fitExponentialDecay([1, 0.1, 100])
    fatigue_analysis.makeVideo("video.mp4", "output.mp4", 0, 2, 150)