import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
class Data:
    def __init__(self, x=None, y=None):
        self._x = x
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
  


