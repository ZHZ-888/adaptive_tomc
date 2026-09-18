#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_following_state_cm(cm, accuracy, free_recall, macro_f1, save_path=None, dpi=600, figsize=(5, 4)):
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    custom_blues = LinearSegmentedColormap.from_list("custom_blues", ["white", "#0000FF"])
    class_names = ["0", "1"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=custom_blues, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted state", fontsize=11)
    ax.set_ylabel("True state", fontsize=11)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})"
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color=text_color)

    metric_text = (
        f"Accuracy    : {accuracy:.3f}\n"
        f"Free recall : {free_recall:.3f}\n"
        f"Macro F1    : {macro_f1:.3f}"
    )

    ax.text(
        0.98, 0.02, metric_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(facecolor="white", edgecolor="black",
                  boxstyle="square,pad=0.25", linewidth=0.8)
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()

#%%
cm = [[1159, 49],
      [94, 3776]]

accuracy = 0.9718393068137062
free_recall = 0.959
macro_f1 = 0.962   # 换成你实际计算出的完整值

save_path = "/home/zzha/PycharmProjects/RampMerging_2026/figures/rf_following_states_norm.pdf"

plot_following_state_cm(cm, accuracy, free_recall, macro_f1, save_path=save_path) # , save_path=save_path