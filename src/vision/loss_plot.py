import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def loss_plot(loss: list):
    x = np.arange(len(loss))
    plt.plot(x, loss)
    plt.title("Loss Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../img/loss_curve/loss_curve.png")
    plt.close()