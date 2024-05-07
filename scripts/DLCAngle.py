import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DLCData:
    def __init__(self, path,fps=30):
        self.data = pd.read_csv(path, skiprows=3, header=None)
        self.fps = fps
        self.length = len(self.data)
        self.process_data()
        self.calculate_time(self.fps)
        print("Extracted data from", path)


    def calculate_angle(self, p1,p2, p3):
        #Calculate the angle at p2
        #Calculate the vectors
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        #Calculate the angle
        angle = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        #Convert to degrees
        angle = np.degrees(angle)
        return angle

    def interpolate(self, df, x, y, likelihood, threshold=0.9):
        for i in range(1, len(df)):
            if df.iloc[i, likelihood] < threshold:
                df.iloc[i, x] = df.iloc[i-1, x]
                df.iloc[i, y] = df.iloc[i-1, y]

    def process_data(self, threshold=0.9, window=30):
        #interpolate
        for i in range(1,len(self.data.columns),3):
            likelihood = i + 2  
            x_col = i  
            y_col = i + 1  
            self.interpolate(self.data, x_col, y_col, likelihood, threshold)

        #moving average
        for i in range(1, len(self.data.columns),3):
            x = self.data.columns[i]
            y = self.data.columns[i+1]
            x_ma = self.data[x].rolling(window=window, min_periods=1).mean()
            y_ma = self.data[y].rolling(window=window, min_periods=1).mean()

            self.data[x] = x_ma
            self.data[y] = y_ma

        #calculate angles
        self.wristAngle = []
        self.mcpAngle = []
        self.pipAngle = []
        for i in range(len(self.data)):  #save angle for each posture for all frames
            forearm = (self.data.iloc[i][self.data.columns[13]], self.data.iloc[i][self.data.columns[14]])
            wrist = (self.data.iloc[i][self.data.columns[10]], self.data.iloc[i][self.data.columns[11]])
            mcp = (self.data.iloc[i][self.data.columns[7]], self.data.iloc[i][self.data.columns[8]])
            pip = (self.data.iloc[i][self.data.columns[4]], self.data.iloc[i][self.data.columns[5]])
            dip = (self.data.iloc[i][self.data.columns[1]], self.data.iloc[i][self.data.columns[2]])
            self.wristAngle.append(self.calculate_angle(forearm, wrist, mcp))
            self.mcpAngle.append(self.calculate_angle(wrist, mcp, pip))
            self.pipAngle.append(self.calculate_angle(mcp, pip, dip))

    def calculate_time(self, fps, slice_interval=10):
        self.time =  np.arange(len(self.data)) / self.fps
        time_labels = [f'{int(t // 60)}:{int(t % 60):02d}' for t in self.time]
        slicing_indices = np.arange(0, len(self.time), slice_interval * 30).astype(int) -1 
        self.labels = [time_labels[i] for i in slicing_indices]
        self.ticks = self.time[slicing_indices]

    def plot(self, slice=None, frameBool=True):
        if slice is None:
            slice = [0,self.length]
        print(slice[0])        
        if not frameBool:
            slice[0] = slice[0].split(':')
            start = (int(slice[0][0]) * 60 + int(slice[0][1])) * self.fps
            slice[1] = slice[1].split(':')
            end = (int(slice[1][0]) * 60 + int(slice[1][1])) * self.fps
        else:
            start = slice[0]
            end = slice[1]
            plt.plot()
            plt.plot(self.time[start:end], self.wristAngle[start:end])
            plt.plot(self.time[start:end], self.mcpAngle[start:end])
            plt.plot(self.time[start:end], self.pipAngle[start:end])
            plt.legend(['Wrist Angle', 'MCP Angle', 'PIP Angle'], fontsize=16)
            plt.xlabel('Frame', fontsize=18)
            plt.ylabel('Angle (degrees)', fontsize=18)
            plt.title('Joint Angles', fontsize=20)
            plt.tick_params(axis='both', labelsize=16)
            plt.show()


if __name__ == '__main__':
    path = "/home/jakejoseph/Desktop/FES_V1-Joseph-2023-10-16/videos/MVI_0401DLC_resnet50_FES_V1Oct16shuffle1_38000.csv"
    data = DLCData(path)
    data.plot()