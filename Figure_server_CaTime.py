
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


# New data provided by the user
data = {
    "No. of nodes": [5, 6, 7, 8, 9, 10],
    "disCa":  [x / 1e3 for x in [2006.670081, 2186.04547, 2315.051581, 2557.011111, 2720.605889, 2956.575189]],
    "SNRCa": [x / 1e3 for x in[2297.503884, 2673.682597, 2994.057944, 3307.768903, 3725.14328, 4081.98063]],
    "SINRCa": [x / 1e3 for x in[2369.93528, 2764.710045, 3187.181771, 3545.429714, 3902.982621, 4216.983481]],
    "distime": [3.504023294, 2.312387572, 1.83038517, 1.187802486, 0.89551453, 0.684278433],
    "SNRtime": [2.157179802, 1.109398019, 0.783679629, 0.597947856, 0.523931241, 0.454334324],
    "SINRtime": [1.374641633, 0.536366568, 0.439568269, 0.198744743, 0.124897307, 0.1161091]
}

df = pd.DataFrame(data)


# Plotting
fig, ax1 = plt.subplots(figsize=(6, 6))
bar_width = 0.5
fontsize = 20
labelsize = 15

# Positions for the bars
positions = np.arange(len(df['No. of nodes'])) * 2.3


ax1.bar(positions - bar_width-0.15, df['disCa'], width=bar_width, label='$\mathcal{C}$ (Direct [9])', color='white', edgecolor='black')
ax1.bar(positions, df['SNRCa'], width=bar_width, label='$\mathcal{C}$ (Multihop [10])', color='gray', edgecolor='black')
ax1.bar(positions + bar_width+0.15, df['SINRCa'], width=bar_width, label='$\mathcal{C}$ (BLSQ)', color='black')

ax1.set_xlabel('No. of Servers', fontsize=fontsize)
ax1.set_ylabel('Network Capacity ($\mathcal{C}$) [Gbps]', fontsize=fontsize)
ax1.set_xticks(positions)
ax1.set_xticklabels(df['No. of nodes'], fontsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_ylim(0, 7)
# ax1.set_xlim(-1.3, 10.3)

# Secondary axis for latency
ax2 = ax1.twinx()
ax2.plot(positions, df['distime'], 'k^--', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (Direct [9])')
ax2.plot(positions, df['SNRtime'], 'bx-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (Multihop [10])')
ax2.plot(positions, df['SINRtime'], 'ro-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (BLSQ)')
ax2.set_ylabel('Network Latency ($\Theta$) [s]', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=labelsize)
latency_ticks = np.arange(0, 5, 1)
ax2.set_ylim(0, 5)
#ax2.set_yticks(latency_ticks)

# Combine legends from both axes
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
ax1.legend(handles, labels, loc='upper left', fontsize=12, ncol = 2, columnspacing=1)

additional_text = "Max. Packet Size = 1500 bytes\nNo. of Devices = 200"
plt.text(0.02, 0.81, additional_text, transform=ax1.transAxes, verticalalignment='top', fontsize=labelsize)


plt.tight_layout()
plt.savefig('Figure/Figure_server_CaTime.pdf', format='pdf', bbox_inches='tight')
