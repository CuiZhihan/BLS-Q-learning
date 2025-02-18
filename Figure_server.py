
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# New data provided by the user
data = {
    "No. of servers": [3, 4, 5, 6, 7, 8, 9, 10],
    "BLSCatime": [x / 1e6 for x in [592167.1812, 616035.7559, 638738.1059, 744873.1954, 822622.9711, 999804.4065, 1128315.995, 1228015.864]],  # Convert from bps to MB/s
    "BLSCadis": [x / 1e6 for x in [970993.1134, 1006089.875, 1103038.533, 1219176.984, 1270508.86, 1681117.06, 1773936.077, 1989265.378]],  # Convert from bps to MB/s
    "BLSCaSINR": [x / 1e6 for x in [1076039.565, 1114933.254, 1242407.568, 1266064.821, 1407958.288, 1777979.676, 1965848.551, 2147800.972]],
    "BLSLatime": [5.053264037, 3.212042866, 2.239284359, 1.906807538, 1.738710593, 1.563994926, 1.29889623, 1.151341509],
    "BLSLadis": [3.068462555, 2.04426903, 1.378858753, 1.174889393, 0.988330991, 0.91590318, 0.810457796, 0.741473893],
    "BLSLaSINR": [2.768909347, 1.744128834, 1.194736986, 0.947430896, 0.733680057, 0.64636211, 0.573908483, 0.531313394]
}

df = pd.DataFrame(data)

# Plotting
fig, ax1 = plt.subplots(figsize=(7, 7))
bar_width = 0.35
fontsize = 20
labelsize = 15

# Positions for the bars
positions = np.arange(len(df['No. of servers'])) *1.5

ax1.bar(positions - bar_width-0.1, df['BLSCatime'], width=bar_width, label='$\mathcal{C}$ (BLS/MT)', color='white', edgecolor='black')
ax1.bar(positions, df['BLSCadis'], width=bar_width, label='$\mathcal{C}$ (BLS/NS)', color='gray', edgecolor='black')
ax1.bar(positions + bar_width+0.1, df['BLSCaSINR'], width=bar_width, label='$\mathcal{C}$ (BLS/MS)', color='black')

ax1.set_xlabel('No. of MEC servers', fontsize=fontsize)
ax1.set_ylabel('Average network capacity ($\mathcal{C}$) [Mbps]', fontsize=fontsize)
ax1.set_xticks(positions)
ax1.set_xticklabels(df['No. of servers'], fontsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_ylim(0, 3.5)
ax1.set_xlim(-0.8, 11.3)
# Secondary axis for latency
ax2 = ax1.twinx()
ax2.plot(positions, df['BLSLatime'], 'k^--', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (BLS/MT)')
ax2.plot(positions, df['BLSLadis'], 'bx-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (BLS/NS)')
ax2.plot(positions, df['BLSLaSINR'], 'ro-', markerfacecolor='w', markersize=10, markeredgewidth=2, linewidth=2, label='$\Theta$ (BLS/MS)')
ax2.set_ylabel('Average network latency ($\Theta$) [s]', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=labelsize)
latency_ticks = np.arange(0, 130, 20)
ax2.set_ylim(0, 6.5)
#ax2.set_yticks(latency_ticks)

# Combine legends from both axes
handles, labels = [], []
for ax in [ax1, ax2]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
ax1.legend(handles, labels, loc='upper right', fontsize=labelsize, ncol = 2)

additional_text = "Max. packet size = 1500 bytes\nNo. of devices = 100"
plt.text(0.38, 0.80, additional_text, transform=ax1.transAxes, verticalalignment='top', fontsize=labelsize)

plt.tight_layout()
plt.savefig('Figure/No_of_servers.pdf', format='pdf', bbox_inches='tight')
