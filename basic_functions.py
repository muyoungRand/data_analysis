#%%
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def read_file(filename, channels = [0]):
    """
    Reads MainWin data files to extract scan variables (frequency, amplitude, delay, etc.) and average excited state population

    Python 2 -> Python 3 Update:
        Replaced open option 'rb' as 'rt'. In Python 2, we read the data file in bytes; now, we read it as a text file.

    Single Channel PMT -> Multi-Channel PMT Update:
        Single channel --> Data is in row[1]
        Multi-Channel --> Gates 5, 7, 9 correspond to row[32], row[34], row[36] correspondingly
    """
    if len(channels) == 1:
        with open(filename, 'rt') as file:
            scan_variable = [] # Variable scanned
            avg_excited_state_population = [] # Chn 5 excited state pop
            for row in csv.reader(file):
                scan_variable.append(float(row[0]))
                avg_excited_state_population.append(float(row[1]))

        return np.asarray(scan_variable), np.asarray(avg_excited_state_population)

    else:
        with open(filename, 'rt') as file:
            scan_variable = [] # Variable scanned
            avg_excited_state_population_chn5 = [] # Chn 5 excited state pop
            avg_excited_state_population_chn7 = [] # Chn 7 excited state pop
            avg_excited_state_population_chn9 = [] # Chn 9 excited state pop
            for row in csv.reader(file):
                scan_variable.append(float(row[0]))
                avg_excited_state_population_chn5.append(float(row[32]))
                avg_excited_state_population_chn7.append(float(row[34]))
                avg_excited_state_population_chn9.append(float(row[36]))

        return np.asarray(scan_variable), np.asarray(avg_excited_state_population_chn5), np.asarray(avg_excited_state_population_chn7), np.asarray(avg_excited_state_population_chn9)
    
def thermal_ratio_fit(rsb_data, bsb_data):
    """
    Uses the Ratio method to calculate nBar for thermal states

    Args:
        rsb_data (data): RSB delay scan data
        bsb_data (data): BSB delay scan data

    Returns:
        float: Caculated nBar
    """
    ratio = [] # Store ratio values

    for i in range(len(rsb_data)):
        if bsb_data[i] != 0:
            ratio.append(rsb_data[i] / bsb_data[i])
    
    r = np.average(ratio)
    nbar = np.absolute(r / (1 - r))

    return nbar

def rsb_multi_sin_fit(t, p):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 2) # Highest Fock state considered

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:-1]) # Ensure sum of population is 1

    Omega_0 = p[max_n_fit] # Carrier Rabi Frequencyu
    gamma = p[max_n_fit + 1] # Decoherence Rate

    Omega = Omega_0 * np.sqrt(np.linspace(0, max_n_fit-1, max_n_fit)) # RSB Rabi Frequencies for Fock states

    for i in range(max_n_fit):
        res = res + a[i] * np.cos(Omega[i] * t) * np.exp(-gamma * (i + 2)**(0.7) * t) # Calculate excited state population
    
    res = 1.0 / 2 - res / 2.0

    return res

def try_rsb_sine_fit(x, y, pop_guess, Omega_0, gamma, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    if lower_bound == None:
        lower_bound = [0 for i in range(len(pop_guess))]
        lower_bound.append(Omega_0 * 0.5)
        lower_bound.append(0)

    if upper_bound == None:
        upper_bound = [1 for i in range(len(pop_guess))]
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.001)

    p = pop_guess + [Omega_0, gamma]
    func_fit = lambda x, *p: rsb_multi_sin_fit(x, p)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, bounds = (lower_bound, upper_bound))

    return popt

    

# %%
