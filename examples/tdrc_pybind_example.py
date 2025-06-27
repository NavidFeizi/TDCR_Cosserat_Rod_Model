import sys, os
import numpy as np
import tdcr_cpp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import figsize, pgf_with_latex

plt.rcParams.update(pgf_with_latex)


def main():

    robot = tdcr_cpp.TDCR(
        15.467e9,  # E
        5.6e9,  # G
        0.0010,  # radius
        0.0040,  # mass
        1.0,  # length
        1.112e-3,  # tendon offset
    )

    # robot.update_point_force([0.0, 0.0, 0.0])
    tau = np.array([0.0, 10.0, 5.0, 0.0])
    robot.update_initial_guess(tau)
    robot.set_tendon_pull(tau)
    robot.solve_bvp()

    P = np.array(robot.get_backbone()).T  # shape (3, 200)
    base_state = robot.get_base_state()
    tip_pos = robot.get_tip_pos()

    #### Plot the results ####
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 2])
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[1, 0]))
    axs.append(fig.add_subplot(gs[0:3, 1], projection="3d"))  # 3D plot

    axs[0].plot(P[2, :], P[0, :], color="black", label="x")
    axs[0].set_xlabel("z [m]")
    axs[0].set_ylabel("x [m]")
    axs[0].set_title("x")
    axs[0].grid("both")

    axs[1].plot(P[2, :], P[1, :], color="black", label="y")
    axs[1].set_xlabel("z [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("y")
    axs[1].grid("both")

    axs[2].plot3D(P[2, :], P[0, :], P[1, :], label="Backbone")
    axs[2].scatter(0.0, 0.0, 0.0, color="red", label="base")
    axs[2].set_xlabel("Z [m]")
    axs[2].set_ylabel("X [m]")
    axs[2].set_zlabel("Y [m]")
    axs[2].set_title("3D Backbone")
    axs[2].legend()
    axs[2].set_box_aspect([1, 1, 1])

    # Equal aspect ratio: set data limits to be the same
    xdata = P[2, :]
    ydata = P[0, :]
    zdata = P[1, :]

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
        os.path.join(os.path.dirname(__file__), "output", "tdcr_pybind_results.png"),
        format="png",
        dpi=600,
    )
    plt.show(block=True)

    # # Example usage:
    # # tdcr_wrapper = TDCRWrapper(tdcr)
    # # result = tdcr_wrapper([0.0, 10.0, 0.0, 0.0])
    # # print("Result from wrapper:", result)

    # print("Backbone first point:", P[0])
    # print("Tip position:", tip_pos)


if __name__ == "__main__":
    main()
