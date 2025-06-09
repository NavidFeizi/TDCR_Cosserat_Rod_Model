import pandas as pd
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linewidth"] = 0.5
matplotlib.rc("font", family="serif", size=7)
matplotlib.rcParams["text.usetex"] = True


df = pd.read_csv("output/backbone.csv", delimiter=",")

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
axs = []

axs.append(fig.add_subplot(gs[0, 0]))      
axs.append(fig.add_subplot(gs[1, 0], projection='3d'))   

axs[0].plot(df["z"]*1e3, df["y"]*1e3, label="Backbone", color="blue")
axs[0].set_aspect('equal')
axs[0].set_title("Backbone")
axs[0].set_xlabel("Z [mm]")
axs[0].set_ylabel("Y [mm]")
axs[0].grid("both")

axs[1].plot(df["z"]*1e3, df["x"]*1e3, df["y"]*1e3, label="Backbone", color="red")
axs[1].set_aspect('equal')
axs[1].set_title("Backbone 3D")
axs[1].set_xlabel("Z [mm]")
axs[1].set_ylabel("X [mm]")
axs[1].set_zlabel("Y [mm]")
axs[1].grid(True)

plt.show()


