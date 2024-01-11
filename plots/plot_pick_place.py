import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

plt.style.use("plots/subfigure.mplstyle")

values_combined = 100
monitor = np.loadtxt("plots/pick_place_monitor.csv", delimiter=',', skiprows=1,)


x_time_steps_values = monitor[:, 1]
size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
x_time_steps_values = x_time_steps_values[:size]
x_time_steps_values = x_time_steps_values.reshape((int(len(x_time_steps_values) / values_combined), values_combined))
x_time_steps_values = np.sum(x_time_steps_values, axis=1)
x_intervals = [0]
for j in x_time_steps_values:
    new_time_step = x_intervals[-1] + j
    x_intervals.append(new_time_step)
x_intervals = np.array(x_intervals)

# success rate:
successes = monitor[:, 3]
successes = successes[:size]
successes = successes.reshape((int(len(successes) / values_combined), values_combined))
y_success_values = np.average(successes, axis=1)
y_success_values = np.insert(y_success_values, 0, 0)

# deletesome values to make it more spiky
for _ in range(6):
    x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
    # y_reward_values = np.delete(y_reward_values, np.arange(0, y_reward_values.size, 2))
    y_success_values = np.delete(y_success_values, np.arange(0, y_success_values.size, 2))

y_success_values = y_success_values * 100
figs, ax = plt.subplots()
ax.plot(x_intervals, y_success_values, linewidth=1.0)
ax.set_xlabel("Subgoal Environment Steps")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim([-5, 105])

# plt.savefig("plots/finished_pdf_plots/pick_place_success_rate.pdf")
plt.show()