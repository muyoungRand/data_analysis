#%%
import numpy as np
from scipy.optimize import curve_fit
import math

def rabi_freq(nStart, nDelta, LD_param):
    """
    Calculates Rabi Frequency for nStart -> nStart + nDelta

    Args:
        nStart (int): Initial phonon state
        nDelta (int): Change in phonon number
        LD_param (float): Lamb-Dicke Parameter

    Returns:
        float: Rabi frequency
    """
    nEnd = nStart + nDelta

    nSmall = min([nStart, nEnd])
    nBig = max([nStart, nEnd])
    factor2 = np.exp(-0.5 * LD_param**2) * LD_param**(np.absolute(nDelta))
    factor3 = np.sqrt(np.math.factorial(nSmall)/np.math.factorial(nBig))
    factor4 = np.sum([((-1)**m) * math.comb(nSmall + np.abs(nDelta), nSmall - m) * (LD_param**m) / np.math.factorial(m) for m in range(nSmall + 1)])

    return factor2 * factor3 * factor4 / LD_param

#%%
def sinsquare_func(t, a, b, Tpi, phase):
    return a + b * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_decay_func(t, a, b, d, Tpi, phase):
    return a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_decay_inverted_func(t, a, b, d, Tpi, phase):
    return 1 - (a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2)

def multi_sin_func(t, p, rsb = True):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 4) # Highest Fock state considered. -4 since there are 4 non-population fit factors

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:-1]) # Ensure sum of population is 1

    Omega_0 = p[max_n_fit] # Carrier Rabi Frequencyu
    gamma = p[max_n_fit + 1] # Decoherence Rate
    amp = p[max_n_fit + 2] # Amplitude factor
    offset = p[max_n_fit + 3] # Offset

    if rsb == True:
        Omega = Omega_0 * np.sqrt(np.linspace(0, max_n_fit-1, max_n_fit)) # RSB Rabi Frequencies for Fock states
    else:
        Omega = Omega_0 * np.sqrt(np.linspace(1, max_n_fit - 1, max_n_fit)) # BSB Rabi Frequencies for Fock states

    for i in range(max_n_fit):
        res = res + a[i] * np.cos(Omega[i] * t) * np.exp(-gamma * (i + 2)**(0.7) * t) # Calculate excited state population
    
    res = amp * (1.0 / 2 - res / 2.0) + offset
    return res

def multi_sin_largeLD_func(t, p):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 5) # Highest Fock state considered

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:-1]) # Ensure sum of population is 1

    Omega_0 = p[max_n_fit] # Carrier Rabi Frequencyu
    gamma = p[max_n_fit + 1] # Decoherence Rate
    amp = p[max_n_fit + 2] # Amplitude factor
    offset = p[max_n_fit + 3] # Offset
    LD_param = p[max_n_fit + 4] # Lamb-Dicke Parameter

    Omega = [Omega_0 * rabi_freq(i, 1, LD_param) for i in range(max_n_fit)]

    for i in range(max_n_fit):
        res = res + a[i] * np.cos(Omega[i] * t) * np.exp(-gamma * (i + 2)**(0.7) * t) # Calculate excited state population

    res = amp * (1.0 / 2 - res / 2.0) + offset
    return res

#%%
def try_multi_sin_fit(x, y, pop_guess, variables, rsb = True, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    Omega_0, gamma, amp, offset = variables

    if lower_bound == None:
        lower_bound = [0 for i in range(max_n_fit)]
        lower_bound.append(Omega_0 * 0.5)
        lower_bound.append(0.0)
        lower_bound.append(0.0)
        lower_bound.append(0.0)

    if upper_bound == None:
        upper_bound = [1 for i in range(max_n_fit)]
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.001)
        upper_bound.append(1.0)
        upper_bound.append(1.0)

    p = pop_guess + variables
    func_fit = lambda x, *p: multi_sin_func(x, p, rsb)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, bounds = (lower_bound, upper_bound))

    if np.sum(popt[:max_n_fit]) > 1.0:
        print("Sum of Fock State Population exceeds 1. Force highest Fock State to 0 population")
        popt[max_n_fit - 1] = 0

    return popt

def try_multi_sin_largeLD_fit(x, y, pop_guess, variables, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    Omega_0, gamma, amp, offset, LD_param = variables

    if lower_bound == None:
        lower_bound = [0 for i in range(max_n_fit)]
        lower_bound.append(Omega_0 * 1/2)
        lower_bound.append(0.0)
        lower_bound.append(0.5)
        lower_bound.append(0.0)
        lower_bound.append(LD_param * 1/2)

    if upper_bound == None:
        upper_bound = [1 for i in range(max_n_fit)]
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.001)
        upper_bound.append(1.0)
        upper_bound.append(0.2)
        upper_bound.append(LD_param * 2)

    p = pop_guess + variables
    func_fit = lambda x, *p: multi_sin_largeLD_func(x, p)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, bounds = (lower_bound, upper_bound))

    if np.sum(popt[:max_n_fit]) > 1.0:
        print("Sum of Fock State Population exceeds 1. Force highest Fock State to 0 population")
        popt[max_n_fit - 1] = 0

    return popt

# %%
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
# %%
