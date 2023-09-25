import os

import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def create_box_plot(
        data: np.ndarray,
        title: str,
        x_label: str,
        x_ticks: list[str],
        y_label: str,
        save_path: str | None,
        show=False,
):
    fig, ax = plt.subplots()
    ax.spines[['bottom', 'top']].set_visible(False)
    ax.boxplot(data.T)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_xticks(
        ticks=np.arange(len(x_ticks)) + 1,
        labels=x_ticks,
        rotation=10,
        rotation_mode='anchor',
    )
    ax.yaxis.grid(True)
    ax.yaxis.grid(True, which='minor', linestyle='--')
    ax.set_ylabel(y_label)
    __save(save_path)
    if show:
        plt.show()
    plt.clf()


def create_grouped_box_plot(
        data_1,
        data_2,
        title: str,
        x_label: str,
        x_ticks: list[str],
        y_label_1: str,
        y_label_2: str,
        save_path: str | None,
        show=False,
):
    positions = np.arange(len(x_ticks))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(x_label)
    ax1.spines[['bottom', 'top']].set_visible(False)
    ax2 = ax1.twinx()

    ax1.boxplot(data_1, widths=0.2, positions=positions - 0.12, boxprops=dict(color='tab:blue'))
    ax1.set_ylabel(y_label_1, color='tab:blue')
    ax2.boxplot(data_2, widths=0.2, positions=positions + 0.12, boxprops=dict(color='tab:orange'))
    ax2.set_ylabel(y_label_2, color='tab:orange')
    ax2.set_xticks(
        ticks=positions,
        labels=x_ticks
    )
    plt.title(title)
    __save(save_path)
    if show:
        plt.show()
    plt.clf()


def create_roc_curve(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        title: str,
        save_path: str | None,
        show=False,
):
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    plt.title(title)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r-')
    ax.axis(xmin=0, xmax=1, ymin=0, ymax=1)
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.spines[['right', 'top', 'bottom', 'left']].set_linestyle('--')
    ax.spines[['right', 'top', 'bottom', 'left']].set_linewidth(0.1)

    ax.grid(linestyle='--', linewidth=0.5, which='both')
    __save(save_path)
    if show:
        plt.show()
    plt.clf()


def __save(save_path):
    if save_path is None:
        return

    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    is_tikz = save_path.endswith('.tikz')
    assert is_tikz or save_path.endswith('.png')
    if is_tikz:
        tikzplotlib.save(save_path)
    else:
        plt.savefig(save_path)
