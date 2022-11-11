# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import find_peaks

''' Clean-Up Functions - Do not touch if you don't know what they do '''

def sort_key(mode):
    """
    Function to sort the n-Order sidebands in increasing frequency order.
    """
    return mode[1]

def duplicate_remover(mode):
    """
    Removes 
        1) First order sidebands
        2) Second order sidebands that cancle out (E.g -R1 + R1)
    """
    removed_first_order = []
    for i in mode:
        if len(i[0]) > 4: # Only first order sidebands have short labels
            removed_first_order.append(i)

    sorted_ls = sorted(removed_first_order) # Sorting puts repeated elements next to each other

    removed_duplicates = []
    i = 0
    while i < len(sorted_ls)-1: 
        if sorted_ls[i][0] == sorted_ls[i+1][0]: # Check that repeated elements are next to each other, then just pick one of them
            removed_duplicates.append(sorted_ls[i])
            i += 2
        else:
            removed_duplicates.append(sorted_ls[i])
            removed_duplicates.append(sorted_ls[i+1])
            i += 1

    ### NOTE: There may be a caveat here so that we miss some actual modes. Need to double check that this is not the case.

    return removed_duplicates

def mode_subtract(mode1, mode2):
    mode_freq = round(mode1[1] - mode2[1], 6)
    mode_label = mode1[0] + '-' + mode2[0]

    return [mode_label, mode_freq]

def mode_sum(mode1, mode2):
    mode_freq = round(mode1[1] + mode2[1], 6)
    mode_label = mode1[0] + '+' + mode2[0]

    return [mode_label, mode_freq]

def cleanup_labels(mode):
    """
    Function to clean up sideband modes' labels.
    Count how many of each mode appears in higher-order sidebands, then orders them in Ax -> R1 -> R2 and IP -> OP1 -> OP2 order
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
                new_label = "0"

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

    # For sidebands that cancel out (E.G -R1 + R1), we named them '0'. For easy syntax, we get rid of all such '0's in the labels
    translation_table = dict.fromkeys(map(ord, '0'), None)
    newlabel = clean_label_ax + clean_label_r1 + clean_label_r2
    newlabel = newlabel.translate(translation_table)

    return [newlabel, mode[1]]

def freq_diff(mode, threshold = 0.010):
    """
    Calculates the differences between mode frequencies.
    
    Returns the identity and separation of any modes split by less than the threshold amount
    """
    differences = []

    for i in range(len(mode)):
        for j in range(i+1, len(mode)):
            val = mode[i][1] - mode[j][1]

            if np.absolute(val) <= threshold and mode[i][0] != mode[j][0]:
                differences.append([val, mode[i][0], mode[j][0]])

    return differences

def freq_diff_modes(mode1, mode2, xmin, xmax, threshold = 0.010, plot = True):
    differences = []

    mode1 = [item for item in mode1 if item[1] >= xmin and item[1] <= xmax]
    mode2 = [item for item in mode2 if item[1] >= xmin and item[1] <= xmax]

    i = 0

    while i < len(mode1):
        for j in range(len(mode2)):
            val = mode1[i][1] - mode2[j][1]

            if np.absolute(val) <= 0.010 and mode1[i][0] != mode2[j][0]:
                differences.append([val, mode1[i][0], mode2[j][0]])
        
        i += 1

    if plot == True:
        print("Overlapping 1st-2nd Order Peaks:")
        for i in differences:
            print('\tMode 1:', i[1], ", Mode 2:", i[2], ", Separtion =", np.round(i[0]*10**(3),1), 'kHz\n')

    return differences

''' Frequency Calculations - 3 Ion Mode Frequencies + Higher Order Sidebands'''
def calc_mode_freq(fax, fr1, fr2):
    '''
    Calculates the theoratical mode frequencies given the lowest-frequency mode frequencies. 
    Refer to https://www-nature-com.libproxy1.nus.edu.sg/articles/s41467-018-08090-0#Sec16 for details
    '''
    fax_1 = ['Ax_IP', fax]
    fax_2 = ['Ax_OP1', np.sqrt(3) * fax]
    fax_3 = ['Ax_OP2', np.sqrt(29/5) * fax]

    fr1_1 = ['R1_IP', fr1]
    fr1_2 = ['R1_OP1', np.sqrt(fr1**2 - fax**2)]
    fr1_3 = ['R1_OP2', np.sqrt(fr1**2 - (12/5)*fax**2)]

    fr2_1 = ['R2_IP', fr2]
    fr2_2 = ['R2_OP1', np.sqrt(fr2**2 - fax**2)]
    fr2_3 = ['R2_OP2', np.sqrt(fr2**2 - (12/5)*fax**2)]

    return [fax_1, fax_2, fax_3, fr1_1, fr1_2, fr1_3, fr2_1, fr2_2, fr2_3]

def first_order(f0, trap_freq):
    """
    Generate first-order sidebands
    """
    sb = []
    for f in trap_freq:
        sb.append(mode_sum(f0, f) )
        sb.append(mode_subtract(f0, f) )

    sb.sort(key = sort_key)
    return sb

def second_order(f0, trap_freq):
    """
    Generate 2nd order sidebands via repeated application of first_order function.
    """
    sb1 = first_order(f0, trap_freq)
    for i in range(1):
        sb2 = []
        for fc in sb1:
            sb2.append(first_order(fc, trap_freq))

    uncleansb2 = [val for sublist in sb2 for val in sublist] #Remove outer brackets to form 1D list

    # Clean the labels up first before removing first orders and duplicates
    cleansb2= []
    for i in uncleansb2:
        supp = cleanup_labels(i)
        cleansb2.append(supp)
    # Remove first orders and duplicates
    cleansb2.sort()
    sb2_removed = duplicate_remover(cleansb2)

    return sb2_removed

def third_order(f0, trap_freq):
    """
    Generate 3rd order sidebands via use of second_order function

    NOT IMPLEMENTED PROPERLY
    """
    sb2 = second_order(f0, trap_freq)
    for i in range(2):
        sb3 = []
        for fc in sb2:
            sb3.append(second_order(fc, trap_freq))
    
    sb3 = [val[0] for sublist in sb3 for val in sublist] 
    sb3 = list(dict.fromkeys(sb3))   
    sb3 = [[sb3[i], sb3[i+1]] for i in range(0, len(sb3), 2)]   
    return sb3


''' Data Analysis and Plotting Functions '''

def first_order_modes(mode, height, xmin, xmax, plot = False):
    '''
    Plot all first-order modes for 3-ion case.
    Print all first-order modes for 3 ions that may be overlapping
    '''
    first_order_labels = ['-1Ax_IP', '-1Ax_OP1', '-1Ax_OP2', '-1R1_IP', '-1R1_OP1', '-1R1_OP2', '-1R2_IP', '-1R2_OP1', '-1R2_OP2', '+1Ax_IP', '+1Ax_OP1', '+1Ax_OP2', '+1R1_IP', '+1R1_OP1', '+1R1_OP2', '+1R2_IP', '+1R2_OP1', '+1R2_OP2']

    modes = [item for item in mode if item[1] >= xmin and item[1] <= xmax] #Remove all elements with frequencies beyond plot range
    
    freq = [modes[i][1] for i in range(len(modes))]
    labels = [modes[i][0] for i in range(len(modes))]

    problems = freq_diff(modes)

    if plot == True:
        for i in range(len(freq)):
            if labels[i] in first_order_labels:
                plt.vlines(freq[i], 0, height, color = 'r')
                plt.text(freq[i]-0.03, height+0.05, labels[i] + '(' + str(round(f0[1] - freq[i], 4)) + ')', rotation = 90, fontsize = 'x-small', color = 'r')
    
        print("First Order Overlapping Peaks:")
        for i in problems:
            print('\tMode 1:', i[1], ", Mode 2:", i[2], ", Separtion =", np.round(i[0]*10**(3),1), 'kHz\n')

    return problems

def second_order_modes_1ion(mode, height, xmin, xmax, plot = False):
    '''
    Plots 1-ion second order modes -> THe 3-ion second order modes are a complete mess.
    Prints any 3-ion modes that may be overlapping
    '''
    modes = [item for item in mode if item[1] >= xmin and item[1] <= xmax] #Remove all elements with frequencies beyond plot range
    
    freq = [modes[i][1] for i in range(len(modes))]
    labels = [modes[i][0] for i in range(len(modes))]

    problems = freq_diff(modes)

    if plot == True:
        for i in range(len(freq)):
            if 'OP1' not in labels[i] and 'OP2' not in labels[i]:
                plt.vlines(freq[i], 0, height, color = 'b')
                plt.text(freq[i]+0.01, height+0.02, labels[i] + '(' + str(round(f0[1] - freq[i], 4)) + ')', rotation = 90, fontsize = 'x-small', color = 'b')

        print("Second Order Overlapping Peaks:")
        for i in problems:
            if i[1] != i[2]:
                print('\tMode 1:', i[1], ", Mode 2:", i[2], ", Separtion =", np.round(i[0]*10**(3),1), 'kHz\n')

    return problems
#%%
plot = True

xmin = 111.2
xmax = 113.3


#plt.plot(freq, chn5, 'black')
#plt.plot(freq[peaks], chn7[peaks], 'rx')

#for i in peaks:
#    if xmin < freq[i] < xmax:
#        plt.text(freq[i] + 0.01, chn5[i]-0.1, str(round(f0[1] - freq[i], 4)), rotation = 90, fontsize = 'x-small')

f0 = ['', 113.20]
ax = 0.462
r1 = 1.55
r2 = 2.01

if plot == True:
    plt.figure(dpi = 200)
    plt.xlim(xmin, xmax)
    plt.axvline(x = f0[1])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel("P(excited)")

valid_modes = []


guess_mode_freq = calc_mode_freq(0.462, r1, r2)

uncleansb1 = first_order(f0, guess_mode_freq)
sb1 = []
for i in uncleansb1:
    sup = cleanup_labels(i)
    sb1.append(sup)

sb2 = second_order(f0, guess_mode_freq)

""" uncleansb3 = third_order(f0, guess_mode_freq)
sb3 = []
for i in uncleansb3:
    supp = cleanup_labels(i)
    sb3.append(supp) """

first_order_overlaps = first_order_modes(sb1, 0.8, xmin, xmax, plot)
second_order_overlaps = second_order_modes_1ion(sb2, 0.4, xmin, xmax, plot)
first_second_order_overlaps = freq_diff_modes(sb1, sb2, xmin, xmax, plot)


#plt.rcParams["figure.figsize"] = (5, 5)
#plot_counter_3ion = 0
#plot_counter_3ion = plot_3ion_modes(sb1, 0.8, xmin, xmax, plot_counter_3ion)
#plot_counter_3ion = plot_3ion_modes(sb2, 0.5, xmin ,xmax, plot_counter1_3ion)

#plot_counter_1ion = 0
#plot_counter_1ion = plot_1ion_modes(sb1, 1.0, xmin, xmax, plot_counter_1ion)
#plot_counter_1ion = plot_1ion_modes(sb2, 0.6, xmin, xmax, plot_counter_1ion)

#plt.show() 

#%%
#freq, chn5, chn7, chn9 = read_file('freqscan globalraman 200us RepLock_-10dBm NoLock_5Amp 123p5_126p5.txt')
#peaks, _ = find_peaks(chn5, height = 0.2, distance = 20, prominence = 0.08)
plot = False

xmin = 111.2
xmax = 113.3


#plt.plot(freq, chn5, 'black')
#plt.plot(freq[peaks], chn7[peaks], 'rx')

#for i in peaks:
#    if xmin < freq[i] < xmax:
#        plt.text(freq[i] + 0.01, chn5[i]-0.1, str(round(f0[1] - freq[i], 4)), rotation = 90, fontsize = 'x-small')

f0 = ['', 113.20]
ax = 0.462
r1 = [1.11 - 0.01 * i for i in range(-50, 50)]
r2 = [1.57 - 0.01 * i for i in range(-50, 50)]


if plot == True:
    plt.figure(dpi = 200)
    plt.xlim(xmin, xmax)
    plt.axvline(x = f0[1])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel("P(excited)")

valid_modes = []

for k in range(len(r1)):
    guess_mode_freq = calc_mode_freq(0.462, r1[k], r2[k])

    uncleansb1 = first_order(f0, guess_mode_freq)
    sb1 = []
    for i in uncleansb1:
        sup = cleanup_labels(i)
        sb1.append(sup)

    sb2 = second_order(f0, guess_mode_freq)

    """ uncleansb3 = third_order(f0, guess_mode_freq)
    sb3 = []
    for i in uncleansb3:
        supp = cleanup_labels(i)
        sb3.append(supp) """

    first_order_overlaps = first_order_modes(sb1, 0.8, xmin, xmax, plot)
    second_order_overlaps = second_order_modes_1ion(sb2, 0.4, xmin, xmax, plot)
    first_second_order_overlaps = freq_diff_modes(sb1, sb2, xmin, xmax)

    if len(first_order_overlaps) == 0 and len(first_second_order_overlaps) < 3 and len(second_order_overlaps) < 10:
        valid_modes.append([r1[k], r2[k]])

print("Possible Modes:", valid_modes)


# %%
