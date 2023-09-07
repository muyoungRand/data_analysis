#%%
import numpy as np
from scipy.optimize import curve_fit
import math

#%% ------------------ Rabi Frequencies & Distributions ------------------
def rabi_freq(nStart, nDelta, LD_param):
    """
    Calculates Rabi Frequency for nStart -> nStart + nDelta

    Args:
        nStart (int): Initial phonon state
        nDelta (int): Change in phonon number
        LD_param (float): Lamb-Dicke Parameter

    Returns:
        float: Rabi frequency normalised to carrier Rabi frequency value
    """
    nEnd = nStart + nDelta

    nSmall = min([nStart, nEnd])
    nBig = max([nStart, nEnd])
    factor2 = np.exp(-0.5 * LD_param**2) * LD_param**(np.absolute(nDelta))
    factor3 = np.sqrt(np.math.factorial(nSmall)/np.math.factorial(nBig))
    factor4 = np.sum([((-1)**m) * math.comb(nSmall + np.abs(nDelta), nSmall - m) * (LD_param**m) / np.math.factorial(m) for m in range(nSmall + 1)])

    return factor2 * factor3 * factor4

def coherent_distribution(nBar, max_n_fit):
    return [nBar**n * np.exp(-nBar) / np.math.factorial(n) for n in range(max_n_fit)]

def thermal_distribution(nBar, max_n_fit):
    return [nBar**n / ((nBar + 1)**(n+1)) for n in range(max_n_fit)]

#%% ------------------ Fourier Transform Fit Equations ------------------
def sinsquare_func(t, a, b, Tpi, phase):
    return a + b * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_decay_func(t, a, b, d, Tpi, phase):
    return a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_decay_inverted_func(t, a, b, d, Tpi, phase):
    return 1 - (a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2)

def multi_sin_func(t, p, rsb = True):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 5) # Highest Fock state considered. -4 since there are 4 non-population fit factors

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:-1]) # Ensure sum of population is 1

    Omega_0 = p[max_n_fit] # Carrier Rabi Frequencyu
    gamma = p[max_n_fit + 1] # Decoherence Rate
    amp = p[max_n_fit + 2] # Amplitude factor
    offset = p[max_n_fit + 3] # Offset
    LD_param = p[max_n_fit + 4] # Lamb-Dicke Parameter

    if rsb == True:
        Omega = Omega_0 * LD_param * np.sqrt(np.linspace(0, max_n_fit-1, max_n_fit)) # RSB Rabi Frequencies for Fock states
    else:
        Omega = Omega_0 * LD_param * np.sqrt(np.linspace(1, max_n_fit - 1, max_n_fit)) # BSB Rabi Frequencies for Fock states

    for i in range(max_n_fit):
        res = res + a[i] * np.cos(Omega[i] * t) * np.exp(-gamma * (i + 2)**(0.7) * t) # Calculate excited state population
    
    res = amp * (1.0 / 2 - res / 2.0) + offset
    return res

def multi_sin_largeLD_func(t, p):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 5) # Highest Fock state considered

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:len(a)]) # Ensure sum of population is 1

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

#%% ------------------ Fitting Functions ------------------
def multi_sin_fit(x, y, pop_guess, variables, rsb = True, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    Omega_0, gamma, amp, offset, LD_param = variables

    if lower_bound == None:
        lower_bound = [0 for i in range(max_n_fit)]
        lower_bound.append(Omega_0 * 1/2)
        lower_bound.append(0.0001)
        lower_bound.append(0.0)
        lower_bound.append(0.02)
        lower_bound.append(LD_param * 1/2)

    if upper_bound == None:
        upper_bound = [1 for i in range(max_n_fit)]
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.003)
        upper_bound.append(1.0)
        upper_bound.append(0.5)
        upper_bound.append(LD_param * 2)

    p = pop_guess + variables
    func_fit = lambda x, *p: multi_sin_func(x, p, rsb)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, bounds = (lower_bound, upper_bound))

    if np.sum(popt[:max_n_fit]) > 1.0:
        print("Sum of Fock State Population exceeds 1. Force highest Fock State to 0 population")
        popt[max_n_fit - 1] = 0

    return popt

def multi_sin_largeLD_fit(x, y, pop_guess, variables, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    Omega_0, gamma, amp, offset, LD_param = variables

    if lower_bound == None:
        lower_bound = [0 for i in range(max_n_fit)]
        lower_bound.append(Omega_0 * 1/2)
        lower_bound.append(0.0001)
        lower_bound.append(0.0)
        lower_bound.append(0.02)
        lower_bound.append(LD_param * 1/2)

    if upper_bound == None:
        upper_bound = [1 for i in range(max_n_fit)]
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.003)
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

def multi_sin_largeLD_coherent_fit(x, y, nBar, max_n_fit, variables, lower_bound = None, upper_bound = None):
    pop_guess = coherent_distribution(nBar, max_n_fit)

    Omega_0, gamma, amp, offset, LD_param = variables

    if lower_bound == None:
        lower_bound = coherent_distribution(nBar, max_n_fit)
        for i in range(len(lower_bound)):
            lower_bound[i] = lower_bound[i] / 2
        lower_bound.append(Omega_0 * 1/2)
        lower_bound.append(0.0001)
        lower_bound.append(0.0)
        lower_bound.append(0.0)
        lower_bound.append(LD_param * 1/2)

    if upper_bound == None:
        upper_bound = coherent_distribution(nBar, max_n_fit)
        for i in range(len(upper_bound)):
            upper_bound[i] = upper_bound[i] * 2
        upper_bound.append(Omega_0 * 2)
        upper_bound.append(0.003)
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

# %% ------------------ Fitting Functions with Least Squares ------------------
def try_multi_sin_largeLD_coherent_fit(x, y, max_n_fit, nBar, variables, lower_bound = None, upper_bound = None):
    res= []
    diff = []
    
    for i in range(-10, 11):
        pop_guess = coherent_distribution(nBar * (100 + i)/100, max_n_fit)

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

        fit_y = multi_sin_largeLD_func(x, popt)

        res.append(popt)
        diff.append(np.linalg.norm(y - fit_y))

    print(diff)
    print(np.argmin(diff))
    return res[np.argmin(diff)]
# %%
