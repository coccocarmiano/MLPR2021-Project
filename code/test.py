import matplotlib.pyplot as plt
import numpy as np
x0, y0 = np.linspace(0, 100), np.linspace(0, 100)
x1, y1 = np.linspace(-50, 50), np.linspace(-50, 50)
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x0, y0)
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x1, y1, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x1, -y1, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x0, -y0, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
axs[0, 0].set_xlabel(None)
axs[0, 1].set_xlabel(None)
axs[0, 1].set_ylabel(None)
axs[1, 1].set_ylabel(None)
plt.tight_layout()
plt.show()