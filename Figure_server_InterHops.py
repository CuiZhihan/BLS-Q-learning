
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


# New data provided by the user
data = {
    "No. of nodes": [5, 6, 7, 8, 9, 10],
    "disInter": [6.659169489, 6.45164071, 5.737123897, 5.097009699, 4.735167694, 4.675641089],
    "SNRInter": [1.881883247, 1.77795609, 1.625651665, 1.459176021, 1.135712816, 0.92646909],
    "SINRInter": [0.844627045, 0.684987553, 0.462074839, 0.250640843, -0.26326273, -0.367153517],
    "disHop" : [2.8296874135, 2.7705232355, 2.769631453, 2.7487918195, 2.675894804, 2.564199647],
    "SNRHop" : [3.098956185, 3.0725321395, 3.016742917, 2.8556125275, 2.771824279, 2.6724921265],
    "SINRHop" : [3.2884478665, 3.192819446, 3.067655413, 2.9827519745, 2.8803631975, 2.8659306575]
}

df = pd.DataFrame(data)


# Plotting
fig, ax1 = plt.subplots(figsize=(6, 6))
bar_width = 0.5
fontsize = 20
labelsize = 15

# Positions for the bars
positions = np.arange(len(df['No. of nodes'])) * 2.3


ax1.bar(positions - bar_width-0.15, df['disHop'], width=bar_width, label='$h$ (Direct [9])', color='white', edgecolor='black')
ax1.bar(positions, df['SNRHop'], width=bar_width, label='$h$ (Multihop [10])', color='gray', edgecolor='black')
ax1.bar(positions + bar_width+0.15, df['SINRHop'], width=bar_width, label='$h$ (BLSQ)', color='black')

ax1.set_xlabel('No. of Servers', fontsize=fontsize)
ax1.set_ylabel('Average Hop Count ($h$)', fontsize=fontsize)
ax1.set_xticks(positions)
ax1.set_xticklabels(df['No. of nodes'], fontsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_ylim(0, 6)
# ax1.set_xlim(-1.3, 10.3)

# Secondary axis for latency
ax2 = ax1.twinx()
ax2.plot(positions, df['disInter'], 'k^--', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (Direct [9])')
ax2.plot(positions, df['SNRInter'], 'bx-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (Multihop [10])')
ax2.plot(positions, df['SINRInter'], 'ro-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (BLSQ)')
ax2.set_ylabel('Network Interference Power ($P_{int}$) [dBm]', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=labelsize)
#latency_ticks = np.arange(0, 600, 100)
ax2.set_ylim(-2, 12)
#ax2.set_yticks(latency_ticks)

# Combine legends from both axes
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
ax1.legend(handles, labels, loc='upper left', fontsize=12, ncol = 2, columnspacing=0.5)

additional_text = "Max. Packet size = 1500 bytes\nNo. of Devices = 200"
plt.text(0.02, 0.81, additional_text, transform=ax1.transAxes, verticalalignment='top', fontsize=labelsize)


plt.tight_layout()
plt.savefig('Figure/Figure_server_InterHops.pdf', format='pdf', bbox_inches='tight')
