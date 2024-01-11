import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
plt.style.use("plots/subfigure.mplstyle")

number_rows = 500000
values_combined = 2000
monitor1 = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor2 = np.loadtxt("plots/mt10/mt10_monitor_2.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor = np.concatenate([monitor1, monitor2])

# monitor = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2, max_rows=number_rows)


fig, ax = plt.subplots(constrained_layout=True)

# time steps:
x_time_steps_values = monitor[:, 1]
size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
x_time_steps_values = x_time_steps_values[:size]
x_time_steps_values = x_time_steps_values.reshape((int(len(x_time_steps_values)/values_combined), values_combined))
x_time_steps_values = np.sum(x_time_steps_values,axis=1)
x_intervals = [0]
for j in x_time_steps_values:
    new_time_step = x_intervals[-1] + j
    x_intervals.append(new_time_step)
x_intervals = np.array(x_intervals)


monitor = monitor[:size]
# success rate:
successes = monitor[:, 4]
successes = successes.reshape((int(size/values_combined), values_combined))
average_success_rates = np.average(successes, axis=1)
average_success_rates = np.insert(average_success_rates, 0, 0)


# weigthed success rate:
non_weighted_average_success_rates = []
for m in range(int(len(monitor)/values_combined)):
    avg_s_rate = 0
    split = monitor[m*values_combined:(m+1)*values_combined]
    for i in range(10):
        filter_tasks = split[:, 3] == i
        successes = split[:, 4][filter_tasks]
        average_success_rate_task = np.average(successes, axis=0)
        avg_s_rate += average_success_rate_task
    non_weighted_average_success_rates.append(avg_s_rate / 10)

non_weighted_average_success_rates.insert(0, 0)


# # delete some values to make it more spiky
for _ in range(5):
    x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
    non_weighted_average_success_rates = np.delete(non_weighted_average_success_rates, np.arange(0, len(non_weighted_average_success_rates), 2))
    average_success_rates = np.delete(average_success_rates, np.arange(0, len(average_success_rates), 2))

non_weighted_average_success_rates = non_weighted_average_success_rates * 100
average_success_rates = average_success_rates * 100
ax.plot(x_intervals, non_weighted_average_success_rates, linewidth=1.0 , label="Equally Weighted Success Rate")
ax.plot(x_intervals, average_success_rates, linewidth=1.0, label="Success Rate")
ax.set_xlabel("Subgoal Environment Steps")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim([-5, 105])
plt.legend(loc="lower right")

plt.savefig("plots/finished_pdf_plots/mt10_success_rate2.pdf")
plt.show()
