import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

No_of_Node = [50, 100, 150]


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'WPA-LEACH', 'DFA-LEACH', 'ROA-LEACH ', 'COA-LEACH', 'PU-HRCOA-LEACH']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(len(No_of_Node)):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('---------------------------------------No of Node - ', i + 1, 'Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('No_of Node-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def plot_results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'Accuracy']
    Algorithm = ['WPA-AMLPNet', 'DFA-AMLPNet', 'ROA-AMLPNet ', 'COA-AMLPNet', 'PU-HRCOA-AMLPNet']
    Classifier = ['SVM', 'Adaboost', 'Random Forest', 'MLP', 'PU-HRCOA-AMLPNet']
    Graph_Terms = [0, 1, 2, 3, 5]
    Epoch = [20, 40, 60, 80]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j]]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Algorithm  Comparison of Epochs')
            X = np.arange(len(Epoch))
            width = 0.08
            ax.bar(X + 0.00 + width / 4, Graph[:, 0], color='darkgreen', edgecolor='w', width=0.15,
                   label=Algorithm[0])
            ax.bar(X + 0.15 + width / 4, Graph[:, 1], color='darkorange', edgecolor='w', width=0.15, label=Algorithm[1])
            ax.bar(X + 0.30 + width / 4, Graph[:, 2], color='#9A32CD', edgecolor='w', width=0.15,
                   label=Algorithm[2])
            ax.bar(X + 0.45 + width / 4, Graph[:, 3], color='#A52A2A', edgecolor='w', width=0.15, label=Algorithm[3])
            ax.bar(X + 0.60 + width / 4, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Algorithm[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['darkgreen', 'darkorange', '#9A32CD', '#A52A2A', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.18), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.30, ('20', '40', '60', '80'), fontname="Arial", fontsize=14,
                       fontweight='bold')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Method  Comparison of Epochs')
            X = np.arange(len(Epoch))
            width = 0.08
            ax.bar(X + 0.00 + width / 4, Graph[:, 5], color='m', edgecolor='w', width=0.15,
                   label=Classifier[0])
            ax.bar(X + 0.15 + width / 4, Graph[:, 6], color='y', edgecolor='w', width=0.15, label=Classifier[1])
            ax.bar(X + 0.30 + width / 4, Graph[:, 7], color='#9b5de5', edgecolor='w', width=0.15,
                   label=Classifier[2])
            ax.bar(X + 0.45 + width / 4, Graph[:, 8], color='#218380', edgecolor='w', width=0.15, label=Classifier[3])
            ax.bar(X + 0.60 + width / 4, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['m', 'y', '#9b5de5', '#218380', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.xticks(X + 0.27, ('20', '40', '60', '80'), fontname="Arial", fontsize=14,
                       fontweight='bold')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold',
                       color='#35530a')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            path = "./Results/%s_Mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'Accuracy']
    Algorithm = ['Optimizer', 'WPA-AMLPNet', 'DFA-AMLPNet', 'ROA-AMLPNet ', 'COA-AMLPNet', 'PU-HRCOA-AMLPNet']
    Classifier = ['Optimizer', 'SVM', 'Adaboost', 'Random Forest', 'MLP', 'PU-HRCOA-AMLPNet']
    Graph_Term = np.array([0, 1, 2, 3, 4, 5]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4, 5]
    table_terms = [Terms[i] for i in Table_Terms]
    Optimizer = ['Adam', 'SGD', 'RMSprop', 'Adamax', 'AdaGrad']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Optimizer)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j, Graph_Term[k]])
            print('--------------------------------', table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Optimizer)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            print('-------------------------------', table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def ConvResults_Pred():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fit.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'WPA-AMLPNet', 'DFA-AMLPNet', 'ROA-AMLPNet ', 'COA-AMLPNet', 'PU-HRCOA-AMLPNet']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('---------------------------------------Prediction - Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
    plt.savefig("./Results/Conv_Pred.png")
    plt.show()


if __name__ == '__main__':
    plotConvResults()
    ConvResults_Pred()
    plot_results()
    Table()
