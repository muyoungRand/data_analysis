#%%
from basic_functions import read_file
import numpy as np
import matplotlib.pyplot as plt
#%%
x, y = read_file("delayscan cpmg 3Pi after_correcting_mode_freq")
for i in range(len(x)):
    x[i] = x[i] * 10**(-6)

plt.plot(x, y)
# %%
chi = -np.log(y)
plt.plot(x, chi)
#%%
chiF = np.fft.rfft(chi)

ts = x[0] - x[1]
sr = 1/ts
freq = np.fft.rfftfreq(chi.size, d = ts)

plt.plot(freq[1:]*10**(-3), chiF[1:])
# %%
noiseLs = []
for k in range(len(freq)):
    wait = x[0]
    f = freq[k]
    var = chiF[0]
    nPi = 3

    tau = [wait * i for i in range(nPi+1)]

    sum_elem = [(-1)**i * (np.exp(2 * np.pi * 1j * f * tau[i]) - np.exp(2 * np.pi * 1j * f * tau[i+1])) for i in range(len(tau)-1)]

    y = (1/(2 * np.pi * f)) * (np.sum(sum_elem))

    noise = var / (4 * np.abs(y))
    noiseLs.append(noise)

plt.yscale("log")
plt.plot(freq*10**((-3)), noiseLs)
# %%
