from matplotlib.animation import FFMpegWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import cv2
import sys
# import deeplabcut
# import dlclive
sys.path.append('/home/lucyliuu/Desktop/stim_behavior/stim_behavior/utils')
from utils import calculate_angle

class DLCInterface:
    """
    A class that provides an interface for working with DeepLabCut.

    Args:
        config (str, optional): The path to the DeepLabCut project configuration file. Defaults to None.
        name (str, optional): The name of the project. Defaults to None.
        author (str, optional): The author of the project. Defaults to None.
        video (str, optional): The path to the video used for the project. Defaults to None.
        working_directory (str, optional): The working directory for the project. Defaults to None.
        maxiters (int, optional): The maximum number of iterations for training. Defaults to 30000.
    """

    def __init__(self, config=None, name=None, author=None, video=None, working_directory=None, maxiters=30000):
        if config is None:
            self.config = deeplabcut.create_new_project(
                name,
                author,
                video,
                working_directory
            )
            deeplabcut.label_frames(self.config)
            deeplabcut.create_training_dataset(self.config, augmenter_type='imgaug')
            deeplabcut.train_network(self.config, shuffle=1, displayiters=100, saveiters=1000, maxiters=maxiters)

        self.config = config

    def analyze(self, path, video=False):
        """
        Analyzes the videos specified by the given path using DeepLabCut.

        Args:
            path (str or list): The path to the video(s) to be analyzed. It can be a single video name or a list of video names.
            video (bool, optional): Whether to create labeled videos. Defaults to False.

        Returns:
            None
        """
        deeplabcut.analyze_videos(self.config, path, shuffle=1, save_as_csv=True, videotype='mp4')
        if video:
            deeplabcut.create_labeled_video(self.config, path, videotype='mp4')

    def refine(self, maxiters=30000):
        """
        Refines the labels and trains the DeepLabCut network.

        Args:
            maxiters (int, optional): The maximum number of iterations for training. Defaults to 30000.

        Returns:
            None
        """
        deeplabcut.refine_labels(self.config)
        deeplabcut.merge_datasets(self.config)
        deeplabcut.create_training_dataset(self.config, net_type='resnet_50', augmenter_type='imgaug')
        # TODO: add automated version to change initial weights of the pose_cfg.yaml file to the latest iteration and to auto set max_input_size to 2000 otherwise training won't start
        deeplabcut.train_network(self.config, shuffle=1, displayiters=100, saveiters=1000, maxiters=maxiters)

    def benchmark(self, video, model=None):
        """
        Perform benchmarking on a video using a DeepLabCut model.

        Args:
            video (str): Path to the video file.
            model (str, optional): Path to the DeepLabCut model. If not provided, a new model will be exported using the
                configuration file specified in `self.config`.

        Returns:
            None
        """
        if model is None:
            model = deeplabcut.export_model(cfg_path=self.config, snapshotindex=-1, TFGPUinference=True, make_tar=False)
        dlclive.benchmark_videos(model, video, resize=0.5, pcutoff=0.5,output='benchmark_results')
    
    def plotInferenceTime(self, filepath):
        """
        Plot the inference time of a DeepLabCut model.

        Args:
            filepath (str): The path to the pickle file containing the inference time data.

        Returns:
            None
        """
        df = pd.read_pickle(filepath)
        self.inferenceTimes = df['inference_times'][0]
        plt.plot(1/df['inference_times'][0])
        plt.xlabel('Frame')
        plt.ylabel('Frames Per Second')
        plt.title('Inference Time)')
        plt.show()



class DataLoader:
    """
    This class loads in NHP predicted joint angles and extracts angles and does preprocessing.

    Attributes:
        name (str): The name of the DataLoader instance.
        file (str): The file path of the data file to load.
        threshold (float): The likelihood threshold for linear interpolation.
        distance_threshold (float): Maximum allowed distance change between frames.
        window (int): The window size for moving average.
        type (str, optional): The type of data. Defaults to 'DLC'.
        wrist (numpy.ndarray): The wrist angles.
        index (numpy.ndarray): The index angles.
        x (list): The x coordinates of the points.
        y (list): The y coordinates of the points.
    """

    def __init__(self, name, file, threshold, window, distance_threshold, forearm_wrist_low, forearm_wrist_up, wrist_mcp_low, wrist_mcp_up, mcp_pip_low, mcp_pip_up,
                                 pip_dip_low, pip_dip_up, type='DLC', fps=30):
        self.file = file
        self.name = name
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.forearm_wrist_low = forearm_wrist_low
        self.forearm_wrist_up = forearm_wrist_up
        self.wrist_mcp_low = wrist_mcp_low
        self.wrist_mcp_up = wrist_mcp_up
        self.mcp_pip_low = mcp_pip_low
        self.mcp_pip_up = mcp_pip_up
        self.pip_dip_low = pip_dip_low
        self.pip_dip_up = pip_dip_up

        self.window = window
        self.type = type
        self.fps=fps
        self.wrist, self.index = self.loadData()

        print("Extracted data from", file)

    def loadData(self):
        """
        Load the data from the file, perform linear interpolation and moving average,
        and extract wrist and index angles.

        Returns:
            tuple: A tuple containing the wrist angles and index angles as numpy arrays.
        """
        data = pd.read_csv(self.file, skiprows=3, header=None)
        self.length = len(data)
        self.likelihoods = [] #modified
        #TODO set a lowerbound for distance of two joints
        self.distances = np.zeros((5, len(data)))
        self.joint_distances = np.zeros((4, len(data)))

        def cal_dist(df, x, y, point_index):
            for i in range(1, len(df)):
        # Calculate Euclidean distance between the current and previous points
                     dx = df.iloc[i, x] - df.iloc[i - 1, x]
                     dy = df.iloc[i, y] - df.iloc[i - 1, y]
                     distance = (dx**2 + dy**2)**0.5
                     self.distances[point_index, i] = distance

        #Calculate distance between frames
        for point_index, i in enumerate(range(1, len(data.columns), 3)):
            x_col = i
            y_col = i + 1
            cal_dist(data, x_col, y_col, point_index)


        def cal_joint_dis(data, i):
                forearm = (data.iloc[i][data.columns[1]], data.iloc[i][data.columns[2]])
                wrist = (data.iloc[i][data.columns[4]], data.iloc[i][data.columns[5]])
                mcp = (data.iloc[i][data.columns[7]], data.iloc[i][data.columns[8]])
                pip = (data.iloc[i][data.columns[10]], data.iloc[i][data.columns[11]])
                dip = (data.iloc[i][data.columns[13]], data.iloc[i][data.columns[14]])
                # Calculate distances between adjacent joints for the current frame
                self.joint_distances[0, i] = ((forearm[0] - wrist[0]) ** 2 + (forearm[1] - wrist[1]) ** 2) ** 0.5  # Forearm → Wrist
                self.joint_distances[1, i] = ((wrist[0] - mcp[0]) ** 2 + (wrist[1] - mcp[1]) ** 2) ** 0.5        # Wrist → MCP
                self.joint_distances[2, i] = ((mcp[0] - pip[0]) ** 2 + (mcp[1] - pip[1]) ** 2) ** 0.5           # MCP → PIP
                self.joint_distances[3, i] = ((pip[0] - dip[0]) ** 2 + (pip[1] - dip[1]) ** 2) ** 0.5           # PIP → DIP

        #Calculate joint distances
        for i in range(len(data)):
            cal_joint_dis(data, i)



        def interpolate(df, x, y, likelihood, threshold, distance_threshold, point_index):
                for i in range(1, len(df)):
        # Replace coordinates if likelihood is below threshold or distance exceeds limit
                     if df.iloc[i, likelihood] < threshold or self.distances[point_index, i-1] > distance_threshold:
                         df.iloc[i, x] = df.iloc[i - 1, x]
                         df.iloc[i, y] = df.iloc[i - 1, y]


        # TODO: joint distance comparison
        def interpolate_distance(df, forearm_wrist_low, forearm_wrist_up, wrist_mcp_low, wrist_mcp_up, mcp_pip_low, mcp_pip_up,
                         pip_dip_low, pip_dip_up):
            for i in range(1, len(df)):
        # Forearm → Wrist
                if self.joint_distances[0, i] < forearm_wrist_low or self.joint_distances[0, i] > forearm_wrist_up:
                    df.iloc[i, 4] = df.iloc[i - 1, 4]    
                    df.iloc[i, 5] = df.iloc[i - 1, 5]    
                    df.iloc[i, 1] = df.iloc[i - 1, 1]    
                    df.iloc[i, 2] = df.iloc[i - 1, 2]

        # Wrist → MCP
                if self.joint_distances[1, i] < wrist_mcp_low or self.joint_distances[1, i] > wrist_mcp_up:
                    df.iloc[i, 7] = df.iloc[i - 1, 7]    
                    df.iloc[i, 8] = df.iloc[i - 1, 8]    
                    df.iloc[i, 4] = df.iloc[i - 1, 4]    
                    df.iloc[i, 5] = df.iloc[i - 1, 5]

        # MCP → PIP
                if self.joint_distances[2, i] < mcp_pip_low or self.joint_distances[2, i] > mcp_pip_up:
                    df.iloc[i, 10] = df.iloc[i - 1, 10]  
                    df.iloc[i, 11] = df.iloc[i - 1, 11]  
                    df.iloc[i, 7] = df.iloc[i - 1, 7]    
                    df.iloc[i, 8] = df.iloc[i - 1, 8]    

        # PIP → DIP
                if self.joint_distances[3, i] < pip_dip_low or self.joint_distances[3, i] > pip_dip_up:
                    df.iloc[i, 13] = df.iloc[i - 1, 13]  
                    df.iloc[i, 14] = df.iloc[i - 1, 14]  
                    df.iloc[i, 10] = df.iloc[i - 1, 10]  
                    df.iloc[i, 11] = df.iloc[i - 1, 11]


        # Linear Interpolation based on likelihood threshold
        if self.type == 'DLC':
            step = 1
        else:
            step = 3  # NOTE: This is for sleap since the first 3 columns are not coordinates

        for point_index, i in enumerate(range(step, len(data.columns), 3)):
            likelihood = i + 2
            x_col = i
            y_col = i + 1
            interpolate(data, x_col, y_col, likelihood, self.threshold, self.distance_threshold, point_index)

        interpolate_distance(data, self.forearm_wrist_low, self.forearm_wrist_up, self.wrist_mcp_low, self.wrist_mcp_up, self.mcp_pip_low, self.mcp_pip_up,
                                 self.pip_dip_low, self.pip_dip_up)

        # Moving average
        for i in range(step, len(data.columns), 3):
            x = data.columns[i]
            y = data.columns[i+1]
            data[x] = data[x].rolling(window=self.window, min_periods=1).mean()
            data[y] = data[y].rolling(window=self.window, min_periods=1).mean()

        wristAngle = []
        indexAngle = []
        # self.forearms = []
        # self.wrists = []
        self.x = []
        self.y = []
        for i in range(len(data)):  # save angle for each posture for all frames
            if self.type == 'DLC':
                #forearm = (data.iloc[i][data.columns[13]], data.iloc[i][data.columns[14]])
                #wrist = (data.iloc[i][data.columns[10]], data.iloc[i][data.columns[11]])
                #mcp = (data.iloc[i][data.columns[7]], data.iloc[i][data.columns[8]])
                #pip = (data.iloc[i][data.columns[4]], data.iloc[i][data.columns[5]])
                #dip = (data.iloc[i][data.columns[1]], data.iloc[i][data.columns[2]])

                forearm = (data.iloc[i][data.columns[1]], data.iloc[i][data.columns[2]])
                wrist = (data.iloc[i][data.columns[4]], data.iloc[i][data.columns[5]])
                mcp = (data.iloc[i][data.columns[7]], data.iloc[i][data.columns[8]])
                pip = (data.iloc[i][data.columns[10]], data.iloc[i][data.columns[11]])
                dip = (data.iloc[i][data.columns[13]], data.iloc[i][data.columns[14]])
                # np.diff(dip)
                #forearm_likelihood = data.iloc[i][data.columns[15]]
                #wrist_likelihood = data.iloc[i][data.columns[12]]
                #mcp_likelihood = data.iloc[i][data.columns[9]]
                forearm_likelihood = data.iloc[i][data.columns[3]]
                wrist_likelihood = data.iloc[i][data.columns[6]]
                mcp_likelihood = data.iloc[i][data.columns[9]]
            else:  # sleap indices # NOTE: Eventually need to make a converter function to handle this
                forearm = (data.iloc[i][data.columns[15]], data.iloc[i][data.columns[16]])
                wrist = (data.iloc[i][data.columns[12]], data.iloc[i][data.columns[13]])
                mcp = (data.iloc[i][data.columns[9]], data.iloc[i][data.columns[10]])
                pip = (data.iloc[i][data.columns[6]], data.iloc[i][data.columns[7]])
                dip = (data.iloc[i][data.columns[3]], data.iloc[i][data.columns[4]])
                forearm_likelihood = data.iloc[i][data.columns[17]]
                wrist_likelihood = data.iloc[i][data.columns[14]]
                mcp_likelihood = data.iloc[i][data.columns[11]]
            # Unpack the points into x and y coordinates
            self.x.append([forearm[0], wrist[0], mcp[0], pip[0], dip[0]])
            self.y.append([forearm[1], wrist[1], mcp[1], pip[1], dip[1]])
            self.likelihoods.append((forearm_likelihood + wrist_likelihood + mcp_likelihood) / 3) #modified
            wristAngle.append(calculate_angle(forearm, wrist, mcp))
            indexAngle.append(calculate_angle(wrist, mcp, pip))  # NOTE: This is using PIP as a vertex, some might use DIP
            

        self.likelihoods = np.array(self.likelihoods)
        
        return np.array(wristAngle), np.array(indexAngle)


    def plot(self, slice=None):
            """
            Plots the wrist angle and index angle.

            Args:
                slice (tuple or int, optional): If `slice` is a tuple of start and end frame numbers, it specifies the range of frames to plot.
                    If `slice` is an integer in the mm::ss format, it specifies the time duration to plot.
                    Defaults to None.
            """
            if slice is None:
                slice = [0, self.length]
            start = slice[0]
            end = slice[1]
            plt.plot(self.wrist[start:end])
            plt.plot(self.index[start:end])
            plt.legend(['Wrist Angle', 'Index Angle'], fontsize=16)
            plt.xlabel('Frame', fontsize=18)
            plt.ylabel('Angle (degrees)', fontsize=18)
            plt.title('Wrist Angle', fontsize=20)
            plt.show()

    def plot2DScatter(self):
        colour = np.arange(len(self.wrist)) / 30
        plt.scatter(self.wrist,self.index, c = colour, cmap = 'viridis',s=20)
        plt.colorbar(label = 'Time (s)')
        plt.xlabel("Wrist Angle (degrees)", fontsize=20)
        plt.ylabel("Index Angle (degrees)", fontsize=20)
        plt.title("Wrist vs Index Angle", fontsize=20)
        plt.tick_params(axis='both', labelsize=16)
        plt.show()

class FatigueAnalysis(DataLoader):
    """
    This class represents a specific analysis for the dmpFES paper.
    It inherits from the DataLoader class and provides methods for fatigue analysis.

    Args:
        name (str): The name of the analysis.
        file (str): The file path of the data file.
        threshold (float): The threshold value for peak detection.
        window (int): The window size for moving average.
        type (str, optional): The type of data. Defaults to 'DLC'.
        peakWindow (int, optional): The window size for peak analysis. Defaults to 20.
        height (int, optional): The height threshold for peak detection. Defaults to 130.
        prominence (int, optional): The prominence threshold for peak detection. Defaults to 10.
        width (int, optional): The width threshold for peak detection. Defaults to 10.
        distance (int, optional): The distance threshold for peak detection. Defaults to 10.
    """

    def __init__(self, name, file, threshold, window, distance_threshold, forearm_wrist_low, forearm_wrist_up, wrist_mcp_low, wrist_mcp_up, mcp_pip_low, mcp_pip_up,
                                 pip_dip_low, pip_dip_up, type='DLC', peakWindow=20, height=130, prominence=10, width=10, distance=10):
        super().__init__(name, file, threshold, window, distance_threshold, forearm_wrist_low, forearm_wrist_up, wrist_mcp_low, wrist_mcp_up, mcp_pip_low, mcp_pip_up,
                                 pip_dip_low, pip_dip_up, type)
        print("Forearm wrist low:", self.forearm_wrist_low)
        self.data = self.wrist  # Only using wrist angle for this paper
        self.peakWindow = peakWindow
        self.height = height
        self.prominence = prominence
        self.width = width
        self.distance = distance
        self.getStimAngle()

    def getStimAngle(self):
        """
        This method calculates the stimulus angle for the analysis.
        It detects peaks in the data and filters out any NaN values.
        It also plots the data to visually inspect the detected peaks.
        """
        self.peaks, _ = find_peaks(self.data, height=self.height, prominence=self.prominence, width=self.width, distance=self.distance)

        angleChanges = []
        for i in range(len(self.peaks)):
            max = np.max(self.data[self.peaks[i] - self.peakWindow:self.peaks[i] + self.peakWindow])
            angleChanges.append(max)
        angleChanges = angleChanges
        means = np.array(angleChanges)
        mask = ~np.isnan(means)
        self.filtered_means = means[mask]
        self.filtered_indices = np.arange(len(means))[mask]

        # Plot to visually inspect if peaks were correctly detected
        plt.figure(figsize=(25, 5))
        plt.plot(self.data)
        #plt.scatter(list(range(len(self.data))), self.data, c=self.likelihoods,s=0.3)
        #plt.colorbar()
        plt.plot(self.peaks, self.filtered_means, ".r", label='Peaks')
        # plt.plot(self.likelihoods, label="Likelihood", color="green", linestyle="--")  ##modified
        plt.title(self.name)
        plt.xlabel("Frames")
        plt.ylabel("Absolute Angle (Degrees)")
        plt.show()

    def plotDist(self):
        point_names = ["Forearm", "Wrist", "MCP", "PIP", "DIP"]

        plt.figure(figsize=(15, 8))

        for i, point_name in enumerate(point_names):
           plt.plot(self.distances[i, :], label=point_name)

        plt.legend(fontsize=12)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Distance", fontsize=14)
        plt.title("Distances for Each joint Over Time", fontsize=16)

        plt.show()

    def plotJointDistances(self):
    ##Plots the distances between adjacent joints (forearm → wrist, wrist → MCP, MCP → PIP, PIP → DIP) over time."""
       joint_labels = ["Forearm → Wrist", "Wrist → MCP", "MCP → PIP", "PIP → DIP"]
       plt.figure(figsize=(15, 8))
    
       for i, label in enumerate(joint_labels):
           plt.plot(self.joint_distances[i, :], label=label)
    
       plt.legend(fontsize=12)
       plt.xlabel("Frame", fontsize=14)
       plt.ylabel("Distance", fontsize=14)
       plt.ylim(0, 1000)
       plt.title("Distances Between Adjacent Joints Over Time", fontsize=16)
       plt.show()


    def stimProfiles(self, stimuli, lengths, starts):
        """
        This method plots the stimulus profiles for the given stimuli, lengths, and start frames.

        Args:
            stimuli (list): List of stimulus indices.
            lengths (list): List of stimulus lengths.
            starts (list): List of start frames for each stimulus.
        """
        fig, axes = plt.subplots(1, len(stimuli), figsize=(20, 5), sharex=True, sharey=True)
        axes = axes.flatten()

        def plotSingleStim(stimNum, ax, length, start):
            """
            Helper function to plot a single stimulus.

            Args:
                stimNum (int): The index of the stimulus.
                ax (matplotlib.axes.Axes): The axes object to plot on.
                length (int): The length of the stimulus.
                start (int): The start frame of the stimulus.
            """
            stim = self.peaks[stimNum]
            subset = self.data[stim - start:stim + length]
            ax.plot(subset, color='black')
            ax.set_title(f"Stim {self.filtered_indices[stimNum] + 1}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Absolute Angle (degrees)")

        for i, (stim, length, start) in enumerate(zip(stimuli, lengths, starts)):
            plotSingleStim(stim, axes[i], length, start)
            axes[i].set_ylim(110, 200)
            axes[i].axhline(y=180, color='red', linestyle='--')
            plt.suptitle(self.name)
        plt.ylabel('Absolute Angle (degrees)')
        plt.tight_layout()
        plt.show()

    def fitExponentialDecay(self, guess, removeStartSlice=0):
        """
        This method fits an exponential decay function to the filtered means.

        Args:
            guess (list): Initial guess values for the exponential decay parameters.
            removeStartSlice (int, optional): Number of initial data points to remove. Defaults to 0.
        """
        def exponential_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        popt, _ = curve_fit(exponential_decay, self.filtered_indices[removeStartSlice:], self.filtered_means[removeStartSlice:], maxfev=10000, p0=guess, bounds=bounds)
        self.exponential = exponential_decay(self.filtered_indices[removeStartSlice:], *popt)
        self.popt = popt

    def makeVideo(self, video_file, saveName, stimNum, start, sliceLength):
        """
        This method creates a video with annotations for a specific stimulus.

        Args:
            video_file (str): The file path of the video.
            saveName (str): The file path to save the annotated video.
            stimNum (int): The index of the stimulus.
            start (int): The start frame of the stimulus.
            sliceLength (int): The length of the stimulus slice.
        """
        stim = self.peaks[stimNum]
        startFrame = stim - start
        endFrame = stim + sliceLength

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
                ax.plot(self.x[i], self.y[i], 'o-', color='white', markersize=3, linewidth=5)
                ax.scatter(self.x[i], self.y[i], c='white', s=300, linewidth=3, edgecolors='black', zorder=10)

                x_pos = 0.7  # 80% from the left of the figure
                y_pos = 0.7  # 50% from the bottom of the figure

                rect = patches.Rectangle((x_pos - 0.08, y_pos - 0.05), 0.16, 0.1,
                                         facecolor='gray', alpha=0.5,
                                         transform=ax.transAxes)

                ax.add_patch(rect)

                ax.text(x_pos, y_pos,
                        f'Angle: {self.data[i]:.0f}',
                        color='white', fontsize=12,
                        ha='center', va='center',
                        transform=ax.transAxes)

                fig.suptitle(f'Monkey {self.name[:1]} Stim {stimNum + 1}',
                             ha='center', va='top', fontsize=20, weight='bold', color='white')

                ax.axis('off')
                writer.grab_frame()
                ax.clear()

                i += 1

        cap.release()
        plt.close(fig)

def plotDecay(data1, data2):
    """
    Plot the decay of ECRB fatigue for Monkey R and Monkey N.

    Args:
        data1 (object): Data object for Monkey R.
        data2 (object): Data object for Monkey N.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data1.filtered_indices, data1.filtered_means, label=data1.name, color='#662506', s=20)

    # Slicing this plot since we didn't include the first 6 values in the fit
    plt.plot(data1.filtered_indices[6:], data1.exponential, label=r'Exponential: $\mathregular{%.0f \cdot e^{-%.2f x} + %.0f}$' % (data1.popt[0], data1.popt[1], round(data1.popt[2],-1)), color='#662506', linestyle='--')

    plt.scatter(data2.filtered_indices, data2.filtered_means, label=data2.name, color='#3182bd', s=20)
    plt.plot(data2.exponential, label=r'Exponential: $\mathregular{%d \cdot e^{-%.2f x} + %.0f}$' % (round(data2.popt[0]), data2.popt[1], round(data2.popt[2],-1)), color='#3182bd', linestyle='--')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Stimulus Number", fontsize=14)
    plt.ylabel("Absolute Angle (Degrees)", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.title("ECRB Fatigue Decay", fontsize=16)
    plt.show()
 
def plotDecaySingle(data, remove_outliers=True, z_threshold=3, remove_initial=0):
    """
    Plot the decay of ECRB fatigue for a single dataset, with optional outlier and initial point removal.

    Args:
        data (object): Data object containing the filtered indices, filtered means, and exponential fit results.
        remove_outliers (bool): Whether to remove outliers based on z-scores. Defaults to True.
        z_threshold (float): Z-score threshold for defining outliers. Defaults to 3.
        remove_initial (int): Number of initial data points to remove. Defaults to 0.
    """
    # Remove the specified number of initial points
    filtered_indices = data.filtered_indices[remove_initial:]  # Skip the first 'remove_initial' points
    filtered_means = data.filtered_means[remove_initial:]      # Skip corresponding means
    
    # Optionally remove outliers
    if remove_outliers:
        # Calculate z-scores for filtered means
        mean = np.mean(filtered_means)
        std_dev = np.std(filtered_means)
        z_scores = np.abs((filtered_means - mean) / std_dev)
        
        # Filter data based on z-score threshold
        valid_indices = z_scores < z_threshold
        filtered_indices = filtered_indices[valid_indices]
        filtered_means = filtered_means[valid_indices]
    
    # Update the exponential fit using only valid data
    popt = data.popt
    fitted_exponential = popt[0] * np.exp(-popt[1] * filtered_indices) + popt[2]

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_indices, filtered_means, label=data.name, color='#662506', s=20)
    plt.plot(
        filtered_indices, 
        fitted_exponential, 
        label=r'Exponential: $\mathregular{%.0f \cdot e^{-%.2f x} + %.0f}$' % (data.popt[0], data.popt[1], round(data.popt[2], -1)), 
        color='#662506', 
        linestyle='--'
    )

    # Formatting the plot
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Stimulus Number", fontsize=14)
    plt.ylabel("Absolute Angle (Degrees)", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.title(f"ECRB Fatigue Decay for {data.name}", fontsize=16)
    plt.show()




if __name__ == "__main__":
    test = DataLoader("test", "/home/jakejoseph/Desktop/AllAngles-Jake-2024-07-30/videos/03-05DLC_mobnet_100_AllAnglesJul30shuffle1_700000.csv", 0.9, 40)
    test.plot(slice=[1290, 6000])
