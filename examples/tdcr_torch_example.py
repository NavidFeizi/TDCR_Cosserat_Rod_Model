import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tdcr_torch import TDCR_Robot

from utils import figsize, pgf_with_latex
plt.rcParams.update(pgf_with_latex)

device = torch.device("cpu")
torch.set_num_threads(12)


def main():

    E = 15.467e9  # E
    G = 5.6e9  # G
    radius = 0.0010  # radius
    mass = 0.0040  # mass
    length = 1.0  # length
    tendon_offset = 1.112e-3  # tendon offset
    num_tendons = 2
    integration_steps = 100

    robot = TDCR_Robot(
        E,
        G,
        radius,
        mass,
        length,
        tendon_offset,
        num_tendons,
        device,
        integration_steps,
    )

    tau = np.array([10.0, 8.0])
    robot.set_tendon_pull(tau)
    Y = robot.bvp_solve_scipy()
    # Y = robot.bvp_solve_shooting()

    #### Plot the results ####
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 2])
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[0:3, 1], projection="3d"))  # 3D plot

    axs[0].plot(Y[2, :], Y[0, :], color="black", label="x")
    axs[0].set_xlabel("z [m]")
    axs[0].set_ylabel("x [m]")
    axs[0].set_title("x")
    axs[0].grid("both")

    axs[1].plot(Y[2, :], Y[1, :], color="black", label="y")
    axs[1].set_xlabel("z [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("y")
    axs[1].grid("both")

    axs[2].plot3D(Y[2, :], Y[0, :], Y[1, :], label="Backbone")
    axs[2].scatter(0.0, 0.0, 0.0, color="red", label="base")
    axs[2].set_xlabel("Z [m]")
    axs[2].set_ylabel("X [m]")
    axs[2].set_zlabel("Y [m]")
    axs[2].set_title("3D Backbone")
    axs[2].legend()
    axs[2].set_box_aspect([1, 1, 1])

    # Equal aspect ratio: set data limits to be the same
    xdata = Y[2, :]
    ydata = Y[0, :]
    zdata = Y[1, :]

    max_range = (
        np.array(
            [
                xdata.max() - xdata.min(),
                ydata.max() - ydata.min(),
                zdata.max() - zdata.min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (xdata.max() + xdata.min()) / 2
    mid_y = (ydata.max() + ydata.min()) / 2
    mid_z = (zdata.max() + zdata.min()) / 2

    axs[2].set_xlim(mid_x - max_range, mid_x + max_range)
    axs[2].set_ylim(mid_y - max_range, mid_y + max_range)
    axs[2].set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "output", "tdcr_torch_results.png"),
        format="png",
        dpi=600,
    )
    plt.show(block=True)


if __name__ == "__main__":
    main()
