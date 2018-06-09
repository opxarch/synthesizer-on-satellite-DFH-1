#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

origin_data = []
processed_data = []

sample_rate = 44100
with open("fft_output.txt", 'r') as f:
    origin_data = [float(x) for x in f.readline().split()]
    processed_data = [float(x) for x in f.readline().split()]

y1 = np.array(origin_data)
y2 = np.array(processed_data)
x1 = np.array(range(len(y1)))

x1 = x1 * sample_rate / len(y1) / 2

fig, (ax1) = plt.subplots(1, 1, sharex=True)
ax1.plot(x1, y1, linewidth=1.0, color='darkmagenta')
ax1.plot(x1, y2, linewidth=1.0, color='darkmagenta')
#ax1.set_xlim(0,len(y1));

fig.tight_layout()
plt.show()
