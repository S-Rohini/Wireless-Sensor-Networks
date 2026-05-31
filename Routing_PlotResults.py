import numpy as np
from matplotlib import pyplot as plt


def Cls_Bar(file_name, ylabel, title):
    Eval = np.load(file_name, allow_pickle=True)[0]
    Classifier = ['ROA-LEACH ', 'COA-LEACH', 'PU-HRCOA-LEACH']
    colors = ['#f15bb5', '#4f772d', 'k']
    No_of_Nodes = [50, 100, 150, 200, 250]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Routing Graph of No. of Nodes')
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    X = np.arange(len(No_of_Nodes) - 2)
    bar3 = ax.barh(X + 0.00, Eval[:3, 2], color=colors[0], height=0.26, label=Classifier[0])
    ax.bar_label(container=bar3, size=10, label_type='edge', labels=[f'{x:.2f}' for x in Eval[:3, 2]],
                 fontweight='bold', color='w', padding=-35)
    bar4 = ax.barh(X + 0.26, Eval[:3, 3], color=colors[1], height=0.26, label=Classifier[1])
    ax.bar_label(container=bar4, size=10, label_type='edge', labels=[f'{x:.2f}' for x in Eval[:3, 3]],
                 fontweight='bold', color='w', padding=-35)
    bar5 = ax.barh(X + 0.50, Eval[:3, 4], color=colors[2], height=0.26, label=Classifier[2])
    ax.bar_label(container=bar5, size=10, label_type='edge', labels=[f'{x:.2f}' for x in Eval[:3, 4]],
                 fontweight='bold', color='w', padding=-35)

    # Remove axes outline
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                   in colors]
    plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=9,
               frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
    plt.xlabel(ylabel, fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='blue')
    plt.yticks(X + 0.20, ['50', '100', '150'], fontname="Arial", fontsize=15, fontweight='bold',
               color='blue')
    plt.ylabel('No. of Nodes', fontname="Arial", fontsize=16, fontweight='bold', color='k')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    path = f"./JournalResults/{title}.png"
    # path = f"./Results/{title}.png"
    plt.savefig(path)
    plt.show()


def Routing_PlotResults():
    Cls_Bar('Eenergy_Eff.npy', 'Energy Efficiency (J)', 'Energy_Efficiency')
    Cls_Bar('Energy_Factor.npy', 'Energy Factor (J)', 'Energy_Factor')
    Cls_Bar('Hop_Count.npy', 'HopCount', 'hop_count')
    Cls_Bar('lifetime.npy', 'Network Lifetime (S)', 'lifetime')
    Cls_Bar('PDR.npy', 'Packet Delivery Ratio (%)', 'PDR')
    Cls_Bar('Stability.npy', 'Stability', 'Stability')
    Cls_Bar('Throughput.npy', 'Throughput (bits-Hz)', 'Throughput')


if __name__ == '__main__':
    Routing_PlotResults()
