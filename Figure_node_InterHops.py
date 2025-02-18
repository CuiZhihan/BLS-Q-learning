
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


# New data provided by the user
data = {
    "No. of nodes": [100, 120, 140, 160, 180, 200],
    "disInter": [-4.49173337, -1.644538689, 0.468541845, 1.784593621, 3.41043016, 4.799234725],
    "SNRInter": [-7.508372795, -4.628507545, -2.937756525, -1.358673412, -0.3238113, 0.936693092],
    "SINRInter": [-8.921998859, -6.85109507, -4.596081235, -3.004979816, -1.7281937, -0.270068398],
    "disHop" : [2.365820904, 2.477538495833333, 2.489404810714286, 2.565129608125, 2.567962898333333, 2.6026418345],
    "SNRHop" : [2.481636163, 2.592891533333333, 2.693060050714286, 2.687599235625, 2.635628781111111, 2.650769633],
    "SINRHop" : [2.805943266, 2.869691211666667, 2.852588404285714, 2.792124235625, 2.840886440555555, 2.8486037375]
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
ax1.bar(positions + bar_width+0.15, df['SINRHop'], width=bar_width, label='$h$ (BLS-Q)', color='black')

ax1.set_xlabel('No. of Devices', fontsize=fontsize)
ax1.set_ylabel('Average Hop Count ($h$)', fontsize=fontsize)
ax1.set_xticks(positions)
ax1.set_xticklabels(df['No. of nodes'], fontsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_ylim(0, 5)
# ax1.set_xlim(-1.3, 10.3)

# Secondary axis for latency
ax2 = ax1.twinx()
ax2.plot(positions, df['disInter'], 'k^--', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (Direct [9])')
ax2.plot(positions, df['SNRInter'], 'bx-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (Multihop [10])')
ax2.plot(positions, df['SINRInter'], 'ro-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$P_{int}$ (BLSQ)')
ax2.set_ylabel('Network Interference Power ($P_{int}$) [dBm]', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=labelsize)
#latency_ticks = np.arange(0, 600, 100)
ax2.set_ylim(-9, 9)
#ax2.set_yticks(latency_ticks)

# Combine legends from both axes
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
ax1.legend(handles, labels, loc='upper left', fontsize=11, ncol = 2, columnspacing=1)

additional_text = "Max. Packet Size = 1500 bytes\nNo. of Servers = 10"
plt.text(0.02, 0.82, additional_text, transform=ax1.transAxes, verticalalignment='top', fontsize=labelsize)


plt.tight_layout()
plt.savefig('Figure/Figure_node_InterHops.pdf', format='pdf', bbox_inches='tight')
