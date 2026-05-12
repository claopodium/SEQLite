import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import os

def heatmap(
    array,        # (L, 20)
    wt_seq,       # WT sequence
    wt_value,     # WT phenotype scalar
    chunk_size=100,
):
    """绘图部分由GPT完成"""

    array = np.squeeze(array)
    savepath = "../img/heatmap"
    os.makedirs(savepath, exist_ok=True)

    L= len(array)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

    # =========================
    # 1. 计算 Δ phenotype
    # =========================
    theta_plot = array - wt_value

    # =========================
    # 2. 颜色归一化（以0为中心）
    # =========================
    vmax = np.max(np.abs(theta_plot))
    if vmax == 0:
        vmax = 1e-8

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # =========================
    # 3. chunk
    # =========================
    if chunk_size is None:
        chunks = [(0, L)]
    else:
        chunks = [(i, min(i + chunk_size, L)) for i in range(0, L, chunk_size)]

    # =========================
    # 4. plot
    # =========================
    for i, (start, end) in enumerate(chunks, 1):

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(
            theta_plot[start:end, :].T,
            cmap="PiYG",   # 粉-绿（关键）
            aspect="auto",
            norm=norm
        )

        # =========================
        # x-axis (position)
        # =========================
        positions = np.arange(start + 1, end + 1)
        step = max(1, len(positions) // 10)
        tick_idx = np.arange(0, len(positions), step)

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(positions[tick_idx])

        # =========================
        # y-axis (AA)
        # =========================
        ax.set_yticks(np.arange(len(alphabet)))
        ax.set_yticklabels(list(alphabet))

        # =========================
        # WT 标记（原点 marker）
        # =========================
        for l, aa in enumerate(wt_seq[start:end]):
            if aa in aa_to_idx:
                ax.scatter(
                    l,
                    aa_to_idx[aa],
                    marker="o",     # 原点
                    color="black",
                    s=40,
                    zorder=5
                )

        # =========================
        # labels
        # =========================
        ax.set_xlabel("Position")
        ax.set_ylabel("Amino Acid")
        ax.set_title(f"ΔPhenotype Heatmap {start+1}-{end}")

        # =========================
        # colorbar
        # =========================
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("Δ Phenotype (Mutant - WT)", rotation=-90, labelpad=15)

        fig.tight_layout()

        # 保存
        fig.savefig(f"{savepath}/heatmap_part_{i}_{start+1}_{end}.png", dpi=300)
        plt.close(fig)