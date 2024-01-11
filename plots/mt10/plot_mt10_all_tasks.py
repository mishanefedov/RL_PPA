import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

plt.style.use("plots/subfigure.mplstyle")


# number_rows = 500000
values_combined = 200
monitor1 = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor2 = np.loadtxt("plots/mt10/mt10_monitor_2.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor = np.concatenate([monitor1, monitor2])

# monitor = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2, max_rows=number_rows)
# print(monitor[1])
print("len monitor:",len(monitor))

fig, ax = plt.subplots(5, 2, constrained_layout=True)
titles = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2','button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']
for i in range(10):
    filter_tasks = monitor[:, 3] == i
    # print(i)
    # time steps:
    x_time_steps_values = monitor[:, 1][filter_tasks]
    # print(len(x_time_steps_values))
    size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
    # print(size)
    x_time_steps_values = x_time_steps_values[:size]
    x_time_steps_values = x_time_steps_values.reshape((int(len(x_time_steps_values)/values_combined), values_combined))
    x_time_steps_values = np.sum(x_time_steps_values,axis=1)
    x_intervals = [0]
    for j in x_time_steps_values:
        new_time_step = x_intervals[-1] + j
        x_intervals.append(new_time_step)
    x_intervals = np.array(x_intervals)
    # success rate:
    successes = monitor[:, 4][filter_tasks]
    print(len(successes))
    successes = successes[:size]
    successes = successes.reshape((int(len(successes)/values_combined), values_combined))
    y_success_values = np.average(successes, axis=1)
    y_success_values = np.insert(y_success_values,0,0)


    # deletesome values to make it more spiky
    for _ in range(3):
        x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
        # y_reward_values = np.delete(y_reward_values, np.arange(0, y_reward_values.size, 2))
        y_success_values = np.delete(y_success_values, np.arange(0, y_success_values.size, 2))
    # print(x_intervals)
    # print(y_success_values)
    print(i)
    r = i % 5
    c = i % 2
    print("row and collumn",r,c)
    y_success_values = y_success_values * 100
    ax[r][c].plot(x_intervals, y_success_values, linewidth=1.3)
    ax[r][c].set_xlabel("Subgoal Environment Steps")
    ax[r][c].set_ylabel("Success Rate (%)")
    ax[r][c].title.set_text(titles[i])
    ax[r][c].set_ylim([-5, 105])

fig.set_size_inches(12, 15)
plt.savefig("plots/finished_pdf_plots/mt10_all_tasks.pdf")
plt.show()