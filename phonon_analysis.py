#%%
import os

import numpy as np
import matplotlib.pyplot as plt

from basic_functions import read_file

directory = '/home/randkim/github/data_analysis/'
# %%
'''
Method 1 - RSB/BSB ratio method
'''

def ratio_method(rsb_filename, bsb_filname):
    rsb_variable, rsb_data = read_file(directory + rsb_filename)
    bsb_variable, bsb_data = read_file(directory + bsb_filename)

    ratio = [] # Store ratio values
    zero_data_points = [] # Store indices where BSB data is 0, and thus gives division errors

    for i in range(len(rsb_data)):
        if bsb_data[i] != 0:
            ratio.append(rsb_data[i]/bsb_data[i])
        else:
            zero_data_points.append(i)

    r = np.average(ratio) 
    nbar = np.absolute(r / (1 - r)) # Cheating here; the absolute shouldn't be there

    if len(zero_data_points) != 0:
        variable = rsb_variable[np.arange(len(rsb_variable)) != zero_data_points[:]] # Remove bad division
    else:
        variable = rsb_variable

    fig, axs = plt.subplots(2, figsize = (8, 6))
    fig.tight_layout(pad = 5.0)
    if 'EIT' in rsb_filename:
        fig.suptitle("Ratio Method - nBar after EIT")
    elif 'Dop' in rsb_filename:
        fig.suptitle("Ratio Method - nBar after Doppler")

    axs[0].plot(rsb_variable, rsb_data, label = 'RSB')
    axs[0].plot(bsb_variable, bsb_data, label = 'BSB')
    axs[0].set_title("Delay Scan Data")
    axs[0].set_xlabel("Delay (us)")
    axs[0].set_ylabel("P(excited)")
    axs[0].legend(shadow=True, fancybox=True)

    axs[1].plot(variable, ratio)
    axs[1].set_title("Calculated RSB/BSB Ratio")
    axs[1].set_xlabel("Delay (us)")
    axs[1].set_ylabel("RSB/BSB")

    plt.show()
    print("Ratio Method: nbar =", np.round(nbar, 2))

rsb_filename = 'delayscan R2BSB globalraman afterDop RepLock-8dBm NoLock33dBAtten.txt'
bsb_filename = 'delayscan R2RSB globalraman afterDop RepLock-8dBm NoLock33dBAtten.txt'
ratio_method(rsb_filename, bsb_filename)

rsb_filename = 'delayscan R2RSB globalraman EIT1500us RepLock-8dBm NoLock33dBAtten.txt'
bsb_filename = 'delayscan R2BSB globalraman EIT1500us RepLock-8dBm NoLock33dBAtten.txt'
ratio_method(rsb_filename, bsb_filename)

# %%
