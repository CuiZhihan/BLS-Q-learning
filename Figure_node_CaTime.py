
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


# New data provided by the user
data = {
    "No. of nodes": [100, 120, 140, 160, 180, 200],
    "disCa":  [x / 1e3 for x in [3361.596777, 3253.386281, 2967.362228, 2731.265128, 2312.991862, 2012.3176]],
    "SNRCa": [x / 1e3 for x in[4669.804028, 4611.489589, 4373.88302, 4327.898822, 4223.879355, 4217.534987]],
    "SINRCa": [x / 1e3 for x in[4897.239952, 4764.399576, 4626.431119, 4449.95807, 4319.85371, 4287.652751]],
    "distime": [0.137744413, 0.210453109, 0.331712653, 0.504174796, 0.635283602, 0.743998669],
    "SNRtime": [0.083046309, 0.118605455, 0.201924375, 0.255556231, 0.32632689, 0.422512646],
    "SINRtime": [0.062565735, 0.087965387, 0.152687194, 0.211619807, 0.27941624, 0.376415804]
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

ax1.set_xlabel('No. of Devices', fontsize=fontsize)
ax1.set_ylabel('Network Capacity ($\mathcal{C}$) [Gbps]', fontsize=fontsize)
ax1.set_xticks(positions)
ax1.set_xticklabels(df['No. of nodes'], fontsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_ylim(0, 9)
# ax1.set_xlim(-1.3, 10.3)

# Secondary axis for latency
ax2 = ax1.twinx()
ax2.plot(positions, df['distime'], 'k^--', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (Direct [9])')
ax2.plot(positions, df['SNRtime'], 'bx-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (Multihop [10])')
ax2.plot(positions, df['SINRtime'], 'ro-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (BLSQ)')
ax2.set_ylabel('Network Latency ($\Theta$) [s]', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=labelsize)
latency_ticks = np.arange(0, 1, 0.2)
ax2.set_ylim(0, 1)
#ax2.set_yticks(latency_ticks)

# Combine legends from both axes
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
ax1.legend(handles, labels, loc='upper left', fontsize=12, ncol = 2, columnspacing=1)

additional_text = "Max. Packet Size = 1500 bytes\nNo. of Servers = 10"
plt.text(0.02, 0.77, additional_text, transform=ax1.transAxes, verticalalignment='top', fontsize=labelsize)


plt.tight_layout()
plt.savefig('Figure/Figure_node_CaTime.pdf', format='pdf', bbox_inches='tight')
