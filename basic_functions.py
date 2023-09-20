#%%
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import assoc_laguerre
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

    if nEnd > 0:
        nSmall = min([nStart, nEnd])
        nBig = max([nStart, nEnd])
        factor2 = np.exp(-0.5 * LD_param**2) * LD_param**(np.absolute(nDelta))
        factor3 = np.sqrt(np.math.factorial(nSmall)/np.math.factorial(nBig))
        factor4 = assoc_laguerre(LD_param**2, nSmall, np.absolute(nDelta))
        return factor2 * factor3 * factor4
    else:
        return 0

def coherent_distribution(nBar, max_n_fit):
    return [nBar**n * np.exp(-nBar) / np.math.factorial(n) for n in range(max_n_fit)]

def thermal_distribution(nBar, max_n_fit):
    return [nBar**n / ((nBar + 1)**(n+1)) for n in range(max_n_fit)]

#%% ------------------ Fourier Transform Fit Equations ------------------
def sinsquare_func(t, p):
    a, b, Tpi, phase = p
    return a + b * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_fit(x, y, y_err, p, lower_bound = None, upper_bound = None):
    a, b, Tpi, phase = p

    if lower_bound == None:
        lower_bound = [0, 0, Tpi/2, 0]
    if upper_bound == None:
        upper_bound = [1, 1, Tpi*2, np.pi*2]

    func_fit = lambda x, *p: sinsquare_func(x, p)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, sigma = y_err, absolute_sigma = True, bounds = (lower_bound, upper_bound))
    
    return popt, pcov

def sinsquare_decay_func(t, a, b, d, Tpi, phase):
    return a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2

def sinsquare_decay_inverted_func(t, a, b, d, Tpi, phase):
    return 1 - (a + b * np.exp(-d * t) * np.sin((np.pi * t / (2 * Tpi)) + phase)**2)

def multi_sin_largeLD_func(t, p):
    res = np.zeros_like(t) # Store calculated excited state population
    max_n_fit = int(np.size(p) - 5) # Highest Fock state considered

    a = list(p[:max_n_fit]) # List of population for each Fock state
    a[-1] = 1 - np.sum(a[:len(a)]) # Ensure sum of population is 1

    Omega_0 = p[max_n_fit]
    LD_param = p[max_n_fit + 1]
    gamma = p[max_n_fit + 2] # Decoherence Rate
    amp = p[max_n_fit + 3] # Amplitude factor
    offset = p[max_n_fit + 4] # Offset

    Omega = [Omega_0 * rabi_freq(i, 1, LD_param) for i in range(max_n_fit)]

    for i in range(max_n_fit):
        res = res + a[i] * np.cos(Omega[i] * t) * np.exp(-gamma * (i + 2)**(0.7) * t) # Calculate excited state population

    res = amp * (1.0 / 2 - res / 2.0) + offset
    return res

def multi_sin_largeLD_fit(x, y, y_err, pop_guess, variables, lower_bound = None, upper_bound = None):
    max_n_fit = len(pop_guess)

    Omega_0, LD_param, gamma, amp, offset = variables

    if lower_bound == None:
        lower_bound = [0 for i in range(max_n_fit)]
        lower_bound.append(Omega_0 * 0.9)
        lower_bound.append(LD_param * 0.95)
        lower_bound.append(0.0)
        lower_bound.append(0.0)
        lower_bound.append(0.0)

    if upper_bound == None:
        upper_bound = [1 for i in range(max_n_fit)]
        upper_bound.append(Omega_0 * 1.1)
        upper_bound.append(LD_param * 1.05)
        upper_bound.append(1.0)
        upper_bound.append(1.0)
        upper_bound.append(1.0)

    p = list(pop_guess) + list(variables)
    func_fit = lambda x, *p: multi_sin_largeLD_func(x, p)
    popt, pcov = curve_fit(func_fit, x, y, p0 = p, sigma = y_err, absolute_sigma= True, bounds = (lower_bound, upper_bound))

    if np.sum(popt[:max_n_fit]) > 1.0:
        #print("Sum of Fock State Population exceeds 1. Force highest Fock State to 0 population")
        popt[max_n_fit - 1] = 0

    return popt, pcov

def try_fit(x, y, y_err, max_n_fit, variables):
    Omega_0, LD_param, gamma, amp, offset = variables

    random_pop_guess = np.random.rand(max_n_fit)
    random_pop_guess = random_pop_guess/sum(random_pop_guess)

    popt, pcov = multi_sin_largeLD_fit(x, y, y_err, random_pop_guess, variables)

    new_pop_guess = popt[:11]

    lower_bound = [0 for i in range(max_n_fit)]
    lower_bound.append(Omega_0 * 0.9)
    lower_bound.append(LD_param * 0.95)
    lower_bound.append(0.0)
    lower_bound.append(0.0)
    lower_bound.append(0.0)

    upper_bound = [1 for i in range(max_n_fit)]
    upper_bound.append(Omega_0 * 1.1)
    upper_bound.append(LD_param * 1.05)
    upper_bound.append(1.0)
    upper_bound.append(1.0)
    upper_bound.append(1.0)

    popt, pcov = multi_sin_largeLD_fit(x, y, y_err, new_pop_guess, variables, lower_bound, upper_bound)

    return popt, pcov
