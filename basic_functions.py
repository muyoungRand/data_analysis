#%%
import csv
import numpy as np

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