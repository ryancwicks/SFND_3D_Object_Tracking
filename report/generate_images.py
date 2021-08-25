import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("*.txt")

laser_data = None
cam_data = {}

for a_file in files:
    detector, descriptor, _ = a_file.split("_")
    data = np.loadtxt(a_file, delimiter=",")
    laser_data = data[:, 0]
    cam_data[detector + "_" + descriptor] = data[:, 1]

plt.plot(laser_data, label="LIDAR Reference")
for key, item in cam_data.items():
    plt.plot(item, label=key)

plt.xlabel("Image")
plt.ylabel("TTC (s)")
plt.legend()

ax = plt.axis()
ax = (ax[0], ax[1], -10, 40)
plt.axis(ax)

plt.show(block=True)
