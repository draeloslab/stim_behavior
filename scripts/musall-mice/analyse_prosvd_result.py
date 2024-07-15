import numpy as np
import matplotlib.pyplot as plt

from stim_behavior.utils.utils import *
from stim_behavior.data_manager import DataManager
from stim_behavior.mice_data_loader import MiceDataLoader

def load_features(dm, data_dir):
    data = dm.load(data_dir, ['dQ', 'ld'])
    Q_norm_diff = data['dQ']
    ld = data['ld']

    mean_values = np.mean(ld, axis=0)
    std_values = np.std(ld, axis=0)
    ld_normalized = (ld - mean_values) / std_values
    ld_normalized -= ld_normalized[0]

    return Q_norm_diff, ld_normalized

def load_all_features(dm):
    data_dict = {
        'cam1': {},
        'cam2': {},
        # 'combined': {},
    }

    for key, dict_val in data_dict.items():
        data_dir = f'{output_dir}/{mouse_id}_{date}_{key}'
        feat1, feat2 = load_features(dm, data_dir)
        dict_val['feat1'] = feat1
        dict_val['feat2'] = feat2

    return data_dict

def plot_results(data_all):
    plt.rcParams.update({
        'font.size': 28,  # Controls default text sizes
        'axes.labelsize': 28,  # X and Y axis label sizes
        'axes.titlesize': 28,  # Title font size
        'xtick.labelsize': 24,  # X-axis tick label size
        'ytick.labelsize': 24,  # Y-axis tick label size
        'legend.fontsize': 16  # Legend font size
    })

    def plot_loadings(data_all):
        f1_10 = data_all['cam1']['feat2']
        f1_01 = data_all['cam2']['feat2']
        f1_11 = data_all['cam2']['feat2'] # TODO: change to 'combined'
        
        tx = (100+np.arange(len(f1_10[:, 0])))/15
        
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))

        # Flatten the axs array for easier iteration
        axs_flat = axs.flatten()

        # Loop through subplots
        for i, ax in enumerate(axs_flat):
            b_i = i # basis index
            a1 = 1.05*np.max([f1_10[:, b_i], f1_01[:, b_i], f1_11[:, b_i]]) # translation factor
            a2 = 1.05*np.min([f1_10[:, b_i], f1_01[:, b_i], f1_11[:, b_i]]) # translation factor
            a = a1 - a2
            ax.plot(tx, f1_10[:, b_i], c='y', label='Cam1', linewidth=3)
            ax.plot(tx, a + f1_01[:, b_i], c='b', label='Cam2', linewidth=3)
            ax.plot(tx, 2*a + f1_11[:, b_i], c='g', label='Combined', linewidth=3)

            ax.set_ylim(-0.5*a, 2.5*a)
            ax.yaxis.set_ticks([])
            if i in [2,3]:
                ax.set_xlabel('Time (s)')
            # ax.xaxis.set_ticks([])

            # ax.axvline(x=10, color='#00ac98', linestyle='--')
            # ax.axvline(x=20, color='#dda900', linestyle='--')

            ax.set_title(f'Basis {b_i}')

        plt.suptitle("Loadings",y=0.9)

        plt.tight_layout()

        # plt.savefig("x.svg")
        plt.show()


    plot_loadings(data_all)


if __name__ == "__main__":
    mouse_id = 'mSM49'
    date = '03-Aug-2018'
    output_dir = '/home/sachinks/Data/tmp'

    dm = DataManager()
    data_all = load_all_features(dm)
    plot_results(data_all)