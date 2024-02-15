import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


class Data:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def sigmoid(x, L ,x0, k, b):
        #x input data
        # L scales output from [0,L]
        # x0 is the midpoint
        # k scales the input
        # b is the offset of output to [b,L+b]
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return (y) 

    def fitCurve(self, function = sigmoid):
        p0 = [max(self.y), np.median(self.x),1,min(self.y)] # this is an mandatory initial guess
        popt, pcov = curve_fit(function, self.x, self.y,p0, method='dogbox')
        fit_y = function(self.x, *popt)

        return fit_y, popt
    
    def getTable(self):
        try:
            return (self.params)
        except:
            print('Need to run electrodePlot() first')
            return None
        
    def addSigmoid(self, x, L, x0, k, b):
        pass

    def mulSigmoid(self):
        pass

    def convSigmoid(self):
        pass


class realData(Data):
    def __init__(self, pulseWidth, y_data):
        super().__init__()
        self.x = pulseWidth
        self.y_data = y_data

    def electrodePlotMatt(self):
        fig, axs = plt.subplots(1,5, figsize= (15,5))
        angles= [ 'Index','MRS','Wrist']
        muscles= ['EDC','FDP','EIP',
                'ECRB', 'FCR' ]
        all_params = []
        for j in range(5):
            for i in range(3):
                # try:
                self.y = self.y_data[j][i].flatten()
                fit_y, params = self.fitCurve()
                all_params.append(list(params) + [j,i])
                # except:
                # print('Could not fit Line for ' + muscles[j] + ' ' + angles[i])
                axs[j].scatter(self.x, self.y_data[j][i], label=angles[i])
                axs[j].plot(self.x, fit_y, linestyle = '--')
            axs[j].set_title(muscles[j])
            axs[j].set_xlabel('Pulse Width')
            axs[j].set_ylabel('Angle Change')
        plt.legend()
        fig.suptitle('Electrode Characterization')
        plt.tight_layout()
        plt.show()
        df = pd.DataFrame(all_params, columns = ['L','x0','k','b', 'Muscle', 'Angle'])
        return(df)
    

class ToyData(Data):
    def __init__(self, k=1, x0=0, L=1, b=0, mu=0, std=0.05):
        super().__init__()
        self.k = k
        self.x0 = x0
        self.L = L
        self.b = b
        self.mu = mu
        self.std = std
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = np.linspace(-10, 10, 100)
        noise = np.random.normal(self.mu, self.std, x.shape)
        x_noisy = x + noise

        y = Data.sigmoid(x_noisy, self.L, self.x0, self.k, self.b)
        y_noisy = y + noise

        self.x = x_noisy
        self.y = y_noisy

        return x_noisy, y_noisy
    
    def electrodePlot(self):
        all_params = []
        fig, axs = plt.subplots(4,4, figsize= (15,15))
        for i in range(16):
            row = i // 4  # Calculate the row index
            col = i % 4   # Calculate the column index
            self.mu = np.random.rand()
            self.std = np.random.uniform(0.05, 0.2)
            self.x, self.y = self.generate_data()
            fit_y, params = self.fitCurve()
            axs[row, col].scatter(self.x, self.y, label='Noisy data')
            axs[row, col].plot(self.x, fit_y, 'r-', label='Fitted')
            axs[row, col].set_xlabel('x')
            axs[row, col].set_ylabel('y')
            axs[row, col].set_title('Electrode ' + str(i))
            # print(params)
            all_params.append(params)
        plt.tight_layout()
        plt.show()
        self.params = pd.DataFrame(all_params, columns = ['L','x0','k','b'])  

class AngleDict:
    def __init__(self, start=-0.5, end=0.5, binSize=0.1):
        self.start = start
        self.end = end
        self.binSize = binSize
        self.d = self.createDict()

    def createDict(self):
        self.d = {}
        self.bins = np.arange(self.start, self.end + self.binSize, self.binSize)
        for x in self.bins:
            # index = round(x, 1) + 0
            for y in self.bins:
                # wrist = round(y, 1) + 0
                key = (float(f"{x:.1f}"), float(f"{y:.1f}"))
                self.d[key] = 0
        return self.d

    def dictChecker(self, pose1, pose2):
        for array1,array2 in zip(pose1,pose2):
            for val1, val2 in zip(array1, array2):
                key = (float(f"{val1[0]:.1f}"), float(f"{val2[0]:.1f}"))
                if key in self.d:
                    self.d[key] += 1
                else:
                    print(f'Key: {key} not in dictionary')

    def dictPlot(self):
        heatmap = (np.array(list(self.d.values())).reshape((len(self.bins), len(self.bins))))
        plt.imshow(heatmap, cmap='hot',interpolation='nearest',extent=[self.start, self.end, self.start, self.end])
        cbar = plt.colorbar()
        cbar.locator = ticker.MaxNLocator(integer=True)  #fix to make sure it's only integers
        cbar.update_ticks()
        cbar.set_label('Number of Occurrences')
        plt.title('Heatmap of Index and Wrist Angle')
        plt.xlabel('Change in Index Angle')
        plt.ylabel('Change in Wrist Angle')
        plt.savefig('heatmap.svg', format='svg')
        plt.show()

if __name__ == "__main__":
    # Usage:
    import scipy.io
    mattData = scipy.io.loadmat("C:/Users/Jjmas/OneDrive/Desktop/Research/Anne/Characterization.mat")
    x_data = mattData['x_data']
    y_data = mattData['y_data']
    pulseWidth = x_data[0][0][0]
    test = AngleDict(start=-0.5, end=0.7)
    index, wrist = y_data[:, 0], y_data[:, 2]
    test.dictChecker(index, wrist)
    test.dictPlot()

    # realParams = realData(y_data=y_data, pulseWidth=pulseWidth)
    # realParams.electrodePlotMatt()
  




