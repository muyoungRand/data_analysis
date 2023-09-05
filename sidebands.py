# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import find_peaks
from operator import itemgetter

#%%
''' Math Functions - Do not touch if you don't know what they do '''

def sort_key(mode):
    """Sort sidebands in increasing frequency order.

    Args:
        mode: [label, frequency]

    Returns:
        Mode Label
    """
    return mode[1]

def duplicate_remover(mode_list):
    """
    Removes 
        1) First order sidebands
        2) Repeatred second order sidebands

    Args:
        mode_list: List of [label, frequency]

    Returns:
        removed_duplicates: mode_list without duplicates
    """
    # 1) Remove first order sidebands
    #    These are never duplicated as first order sidebands are as they are
    removed_first_order = []

    for i in mode_list:
        if len(i[0]) > 4: # Only first order sidebands have short labels
            removed_first_order.append(i)

    # 2) Remove second order sidebands that are duplicated
    sorted_ls = sorted(removed_first_order) # Puts repeated elements next to each other

    removed_duplicates = []

    i = 0
    while i < len(sorted_ls)-1: 
        # Check if susbsequent elements are repeated
        if sorted_ls[i][0] == sorted_ls[i+1][0]: 
            # If yes, then only save one of them
            removed_duplicates.append(sorted_ls[i])
            i += 2
        else:
            # If not, save both values
            removed_duplicates.append(sorted_ls[i])
            removed_duplicates.append(sorted_ls[i+1])
            i += 1

    return removed_duplicates

def mode_subtract(mode1, mode2):
    """
    Subtract modes.

    Args:
        mode1: [label_1, frequency_1]
        mode2: [label_2, frequency_2]

    Returns:
        mode_subtracted: [label_1 - label_2, frequency_1 - frequency_2]
    """
    mode_freq = round(mode1[1] - mode2[1], 6)
    mode_label = mode1[0] + '-' + mode2[0]

    return [mode_label, mode_freq]

def mode_sum(mode1, mode2):
    """
    Sum modes.

    Args:
        mode1: [label_1, frequency_1]
        mode2: [label_2, frequency_2]

    Returns:
        mode_subtracted: [label_1 + label_2, frequency_1 + frequency_2]
    """
    mode_freq = round(mode1[1] + mode2[1], 6)
    mode_label = mode1[0] + '+' + mode2[0]

    return [mode_label, mode_freq]

def cleanup_labels(mode):
    """
    Clean up sideband modes' labels.
    Count how many of each mode appears in higher-order sidebands, then orders them in Ax -> R1 -> R2 and IP -> OP1 -> OP2 order

    Args:
        mode: [messy_label, frequency]

    Returns:
        mode: [clean_label, frequency]
    """
    mode_label = mode[0]
    labels_ax = ['Ax_IP', 'Ax_OP1', 'Ax_OP2']
    labels_r1 = ['R1_IP', 'R1_OP1', 'R1_OP2']
    labels_r2 = ['R2_IP', 'R2_OP1', 'R2_OP2']

    clean_label_ax = ""
    clean_label_r1 = ""
    clean_label_r2 = ""
 
    if 'Ax' in mode_label:
        for lab in labels_ax:
            nModes = mode_label.count("+" + lab) - mode_label.count("-" + lab)

            if nModes > 0:
                new_label = "+" + str(nModes) + lab
            elif nModes < 0:
                new_label = str(nModes) + lab
            else:
                new_label = "0" # For sidebands that cancel out (E.G -A1 + A1), we named them '0'. 


            clean_label_ax += new_label
    
    if 'R1' in mode_label:
        for lab in labels_r1:
            nModes = mode_label.count("+" + lab) - mode_label.count("-" + lab)

            if nModes > 0:
                new_label = "+" + str(nModes) + lab
            elif nModes < 0:
                new_label = str(nModes) + lab
            else:
                new_label = "0"

            clean_label_r1 += new_label

    if 'R2' in mode_label:
        for lab in labels_r2:
            nModes = mode_label.count("+" + lab) - mode_label.count("-" + lab)

            if nModes > 0:
                new_label = "+" + str(nModes) + lab
            elif nModes < 0:
                new_label = str(nModes) + lab
            else:
                new_label = "0"

            clean_label_r2 += new_label

    # For easy syntax, we get rid of all '0's in labels
    translation_table = dict.fromkeys(map(ord, '0'), None)
    newlabel = clean_label_ax + clean_label_r1 + clean_label_r2
    newlabel = newlabel.translate(translation_table)

    return [newlabel, mode[1]]

def freq_diff(mode_list, threshold = 0.010):
    """
    Calculates the differences between modes in mode_list
    Returns identity and separation of modes split by less than the threshold amount

    Args:
        mode_list: List of [label, frequency]
        threshold (float, optional): Frequency separations below threshold are returned. Defaults to 0.010.

    Returns:
        differences: [frequency_difference, label_1, label_2]
    """
    differences = []

    # Calculate separation between all modes in mode_list
    for i in range(len(mode_list)):
        for j in range(i+1, len(mode_list)):
            val = mode_list[i][1] - mode_list[j][1]

            if np.absolute(val) <= threshold and mode_list[i][0] != mode_list[j][0]: # Ensure that R1 - R1 = 0 doesn't happen.
                differences.append([val, mode_list[i][0], mode_list[j][0]])

    return differences

def freq_diff_modes(mode1, mode2, xmin, xmax, threshold = 0.010, text = True):
    """
    Calculates differences between two mode frequency lists (E.g - First order and Second order)
    Return identity and separation of modes split by less than the threshold amount

    Args:
        mode1: List of [label, frequency]
        mode2: List of [label, frequency]
        xmin (float): Minimum frequency considered
        xmax (float): Maximum frequency considered
        threshold (float, optional): Frequency separations below threshold are returned. Defaults to 0.010.. Defaults to 0.010.
        text (bool, optional): Option to print return variable. Defaults to True.

    Returns:
        differences: [frequency_difference, label_1, label_2]
    """
    differences = []

    # Only consider mode frequencies between xmin, xmax
    mode1 = [item for item in mode1 if item[1] >= xmin and item[1] <= xmax]
    mode2 = [item for item in mode2 if item[1] >= xmin and item[1] <= xmax]

    i = 0
    while i < len(mode1):
        for j in range(len(mode2)):
            val = mode1[i][1] - mode2[j][1]

            if np.absolute(val) <= 0.030 and mode1[i][0] != mode2[j][0]: # Ensure that R1 - R1 = 0 doesn't happen.
                differences.append([val, mode1[i][0], mode2[j][0]])
        
        i += 1

    if plot == True:
        print("Overlapping 1st-2nd Order Peaks:")
        for i in differences:
            print('\tMode 1:', i[1], ", Mode 2:", i[2], ", Separtion =", np.round(i[0]*10**(3),1), 'kHz\n')

    return differences

#%%
''' Frequency Calculations - For 3-Ion Chain'''

def calc_mode_freq(fax, fr1, fr2):
    """
     Calculates the theoratical mode frequencies given the lowest-frequency mode frequencies. 
    Refer to https://www-nature-com.libproxy1.nus.edu.sg/articles/s41467-018-08090-0#Sec16 for details

    Args:
        fax (float): Axial Single Ion/In-Phase Frequency in terms of separation from carrier
        fr1 (float): Radial l ''
        fr2 (float): Radial 2 ''

    Returns:
        List of Ax/R1/R2_IP/OP1/OP2 Frequencies
    """
    fax_1 = ['Ax_IP', fax]
    fax_2 = ['Ax_OP1', np.sqrt(3) * fax]
    fax_3 = ['Ax_OP2', np.sqrt(29/5) * fax]

    fr1_1 = ['R1_IP', fr1]
    fr1_2 = ['R1_OP1', np.sqrt(fr1**2 - (fax**2))]
    fr1_3 = ['R1_OP2', np.sqrt(fr1**2 - (12/5)*(fax**2))]

    fr2_1 = ['R2_IP', fr2]
    fr2_2 = ['R2_OP1', np.sqrt(fr2**2 - (fax**2))]
    fr2_3 = ['R2_OP2', np.sqrt(fr2**2 - (12/5)*(fax**2))]

    return [fax_1, fax_2, fax_3, fr1_1, fr1_2, fr1_3, fr2_1, fr2_2, fr2_3]

def first_order(f0, trap_freq):
    """
    Generate first-order sidebands

    Args:
        f0 (float): Carrier frequency
        trap_freq (list): List of mode frequencies

    Returns:
        sb (list): List of first order sidebands in absolute frequency terms
    """

    # Unfortunately I can't use the cleanup_labels here as it messes with generation of the second order sidebands.
    # Maybe fix in the future?
    sb = []
    for f in trap_freq:
        sb.append(mode_sum(f0, f) )
        sb.append(mode_subtract(f0, f) )

    sb.sort(key = sort_key)
    return sb

def second_order(f0, trap_freq):
    """
    Generate 2nd order sidebands via repeated application of first_order function.

    Args:
        f0 (float): Carrier frequency
        trap_freq (list): List of mode frequencies

    Returns:
        sb2_removed (list): List of second order sidebands in absolute frequency terms
    """
    # Still don't exactly know why this works; ask Dzmitry
    sb1 = first_order(f0, trap_freq)
    for i in range(1):
        sb2 = []
        for fc in sb1:
            sb2.append(first_order(fc, trap_freq))

    uncleansb2 = [val for sublist in sb2 for val in sublist] #Remove outer brackets

    # Clean the labels up
    cleansb2= []

    for i in uncleansb2:
        supp = cleanup_labels(i)
        cleansb2.append(supp)

    # Remove first orders and duplicates
    cleansb2.sort()
    sb2_removed = duplicate_remover(cleansb2)

    return sb2_removed

#%%
''' Data Analysis and Plotting '''

def first_order_modes(mode, height, xmin, xmax, plot = False):
    """
    Plot all first-order modes for 3 ions.

    Args:
        mode (list): List of first order sidebands
        height (float): Height of vertical lines indicating mode frequencies
        xmin (float): Minimum frequency to plot
        xmax (float): Maximum frequency to plot
        plot (bool, optional): Set True to plot graph. Defaults to False.

    Returns:
        problems (list): List of frequencies and associated modes that are separated less than threshold amount
    """
    first_order_labels = ['-1Ax_IP', '-1Ax_OP1', '-1Ax_OP2', '-1R1_IP', '-1R1_OP1', '-1R1_OP2', '-1R2_IP', '-1R2_OP1', '-1R2_OP2', '+1Ax_IP', '+1Ax_OP1', '+1Ax_OP2', '+1R1_IP', '+1R1_OP1', '+1R1_OP2', '+1R2_IP', '+1R2_OP1', '+1R2_OP2']

    modes = [item for item in mode if item[1] >= xmin and item[1] <= xmax] #Remove all elements with frequencies beyond plot range
    
    freq = [modes[i][1] for i in range(len(modes))]
    labels = [modes[i][0] for i in range(len(modes))]

    problems = freq_diff(modes, threshold = 0.050)

    if plot == True:
        for i in range(len(freq)):
            if labels[i] in first_order_labels:
                plt.vlines(freq[i], 0, height, color = 'r')
                plt.text(freq[i]-0.03, height+0.05, labels[i] + '(' + str(round(f0[1] - freq[i], 4)) + ')', rotation = 90, fontsize = 'x-small', color = 'r')

    return problems

def second_order_modes_1ion(mode, height, xmin, xmax, plot = False):
    """
    Plots all second-order modes for 3 ions
    '''

    Args:
        mode (list): List of first order sidebands
        height (float): Height of vertical lines indicating mode frequencies
        xmin (float): Minimum frequency to plot
        xmax (float): Maximum frequency to plot
        plot (bool, optional): Set True to plot graph. Defaults to False.

    Returns:
        problems (list): List of frequencies and associated modes that are separated less than threshold amount
    """
    modes = [item for item in mode if item[1] >= xmin and item[1] <= xmax] #Remove all elements with frequencies beyond plot range
    
    freq = [modes[i][1] for i in range(len(modes))]
    labels = [modes[i][0] for i in range(len(modes))]

    problems = freq_diff(modes)

    if plot == True:
        for i in range(len(freq)):
            plt.vlines(freq[i], 0, height, color = 'b')
            plt.text(freq[i]+0.01, height+0.02, labels[i] + '(' + str(round(f0[1] - freq[i], 4)) + ')', rotation = 90, fontsize = 'x-small', color = 'b')

    return problems
#%%
''' Search for Optimal Mode Frequencies using a Point system'''
plot = False

f0 = ['', 113.196316]
ax = 0.440
r1 = np.linspace(0.8, 1.2, num = 40)
r2 = np.linspace(1.0, 2.0, num = 100)

valid_modes = []

for k in range(len(r1)):
    for j in range(len(r2)):

        r_avg = (r1[k] + r2[j])/2
        # Check that RF Vpp is not too high or low
        if r_avg > 1.39 or r_avg < 1.0:
            continue
        else:
            guess_mode_freq = calc_mode_freq(ax, r1[k], r2[j])

            uncleansb1 = first_order(f0, guess_mode_freq)
            sb1 = []
            for i in uncleansb1:
                sup = cleanup_labels(i)
                sb1.append(sup)

            sb2 = second_order(f0, guess_mode_freq)

            first_order_overlaps = first_order_modes(sb1, 0.8, 111, 115.5, plot = False)
            first_second_order_overlaps = freq_diff_modes(sb1, sb2, 111, 115.5, text = False)

            if len(first_order_overlaps) == 0: # Don't want ANY first-order overlaps
                points = 0 # Keep track of overall penalty points
                points_freq_low = 0 # No. of times mode 1st-2nd order overlap is <20kHz
                points_freq_high = 0 # No. of times mode 1st_2nd order overlap is <10kHz
                points_ax = 0 # No. of times 'Ax' is involved in 1st-2nd ovrder overlap

                for hi in first_second_order_overlaps:
                    if 0.010 < np.absolute(hi[0]) <= 0.020:
                        # Minor issue, 1 point
                        points += 1
                        points_freq_low += 1
                        points_freq_high += 0
                        points_ax += 0
                    elif np.absolute(hi[0]) <= 0.010:
                        # Major issue, 19 points
                        points += 19
                        points_freq_low += 0
                        points_freq_high += 19
                        points_ax += 0
                    else:
                        points += 0
                        points_freq_low += 0
                        points_freq_high += 0
                        points_ax += 0
                    if 'Ax' in hi[1] or 'Ax' in hi[2]:
                        # Moderate issue, 5 points
                        points += 5
                        points_freq_low += 0
                        points_freq_high += 0
                        points_ax += 5
                    else:
                        points += 0
                        points_freq_low += 0
                        points_freq_high += 0
                        points_ax += 0
            else:
                # For completely invalid modes with first-order overlaps
                points = 1000
                points_freq_low = 1000
                points_freq_high = 1000
                points_ax = 1000

            if points != 1000: # Ignore mode frequencies with a lot of points
                #print("Resonable Vpp", np.round(r1[k],3), np.round(r2[j],3), np.round((r1[k] + r2[j])/2,3))
                valid_modes.append([r1[k], r2[j], points, points_freq_low, points_freq_high, points_ax/5])

# Sort valid modes wrt total points
sorted_valid_modes = sorted(valid_modes, key = itemgetter(2))
# %%
''' Read Data '''
import ast

# read file format after July 15
def read_file(filename):
    with open(filename, 'rt') as file:
        d = []
        h = []
        r = []
        t = []
        for row in csv.reader(file):
            data  = []
            extra = []
            for item in row:
                try:
                    data.append(float(item))
                except:
                    extra.append(ast.literal_eval(item))
            d.append(data)
            t.append(extra[0])
            h.append(extra[1])
            r.append(extra[2])

    return np.transpose(d), h, r, t

fname = "freqscan globalraman 50us EIT1500us RepLock-8dBm NoLockDDS1003 -0p3VSqz.txt"
data, extra1, extra2, extra3 = read_file(fname)

x = data[0]
y1 = data[1]
y5 = data[32]
y7 = data[34]
y9 = data[36]
#%%
plot = True

xmin = 111.4
xmax = 113.2

#plt.plot(freq[peaks], chn7[peaks], 'rx')

#for i in peaks:
#    if xmin < freq[i] < xmax:
#        plt.text(freq[i] + 0.01, chn5[i]-0.1, str(round(f0[1] - freq[i], 4)), rotation = 90, fontsize = 'x-small')

f0 = ['', 112.944652]
ax = 0.2668

check = []
""" for i in sorted_valid_modes:
    if i[2] < 40:
        check.append(i) """

check = []
check.append([0.810, 0.911])

for i in check:
    r1 = i[0]
    r2 = i[1]

    if plot == True:
        plt.figure(figsize = (15,5),dpi = 150)
        plt.xlim(xmin, xmax)
        plt.axvline(x = f0[1])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel("P(excited)")
        plt.title("R1:" + str(np.round(r1,3)) + "\nR2:" + str(np.round(r2,3)), loc = 'left')

        plt.plot(x, y1, 'black')
        
    guess_mode_freq = calc_mode_freq(ax, r1, r2)

    uncleansb1 = first_order(f0, guess_mode_freq)
    sb1 = []
    for i in uncleansb1:
        sup = cleanup_labels(i)
        sb1.append(sup)

    sb2 = second_order(f0, guess_mode_freq)

    print("Ax:", np.round(ax, 3), "R1:", np.round(r1, 3), "R2:", np.round(r2, 3))
    print("----------------------------")
    first_order_overlaps = first_order_modes(sb1, 0.8, xmin, xmax, plot)
    second_order_overlaps = second_order_modes_1ion(sb2, 0.4, xmin, xmax, plot = True)
    #first_second_order_overlaps = freq_diff_modes(sb1, sb2, xmin, xmax, text = True)

    plt.show()

#plt.show() 

# %%
