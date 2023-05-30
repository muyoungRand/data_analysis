#%%
import basic_functions
import matplotlib.pyplot as plt
import glob
import numpy as np

# %%
path = "/mnt/dzmitrylab/experiment/2023/05/30/*"
final_files = []
for file in glob.glob(path, recursive=True):
    if 'Displacement' in file:
        if '.seq' not in file:
            if '.set' not in file:
                final_files.append(file)

final_files = sorted(final_files, key = lambda ele: (ele.isnumeric(), int(ele) if ele.isnumeric() else ele))

#variable = [int(final_files[i][-6:-2]) for i in range(len(final_files))]
variable = final_files

res = []

#%%
for i in range(len(final_files)):
    file = final_files[i]

    x, y = basic_functions.read_file(file)
    pop_guess = [0.1, 0.22, 0.25, 0.2, 0.1, 0.01, 0.01, 0.03, 0.03, 0.03, 0.01]
    Omega_0 = 1 / (2 * np.pi * 10)
    gamma = 1E-5

    fit_res = basic_functions.try_rsb_sine_fit(x , y, pop_guess, Omega_0, gamma, rsb = False)

    max_n_fit = len(pop_guess)    
    if np.sum(fit_res[:max_n_fit]) > 1.0:
        print("Sum of Fock State Population exceeds 1. Force highest Fock State to 0 population")
        fit_res[max_n_fit - 1] = 0

    plt.figure()
    plt.plot(x, y, 'x-', label = 'Data')
    plt.plot(x, basic_functions.rsb_multi_sin_fit(x, fit_res, rsb = False), label = "Multi Sine Fit")
    plt.title(variable[i])
    plt.legend()
    plt.show()

    plt.figure()
    plt.bar([i for i in range(len(pop_guess))], fit_res[:len(pop_guess)], 0.4, label = "Multi Sine Fit")
    plt.title(variable[i])
    plt.legend()

    res.append([fit_res[:max_n_fit], variable[i]])

# %%
fig = plt.figure(figsize=(15, 15))
fig.suptitle("R1 BSB Shake Trap 1000us 1000Amp", fontsize = 30)

ax = fig.add_subplot(111, projection='3d')

for data in range(len(res)):
    ys = res[data][0]

    ax.bar([i for i in range(len(pop_guess))], ys, zs= res[data][1], zdir='y', alpha=0.8)

ax.set_xlabel('Fock State', fontsize = 15)
ax.set_ylabel('Ring Down Duration', fontsize = 15)
ax.set_zlabel('Population', fontsize = 15)

ax.xaxis.set_tick_params(labelsize='large')
ax.yaxis.set_tick_params(labelsize='large')
ax.zaxis.set_tick_params(labelsize='large')

plt.tight_layout()
plt.show()
# %%
