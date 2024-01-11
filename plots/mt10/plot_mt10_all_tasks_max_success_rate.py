import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
seaborn.set(font_scale=1.2)
plt.style.use("plots/subfigure.mplstyle")

number_rows = 50000
values_combined = 200
monitor1 = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor2 = np.loadtxt("plots/mt10/mt10_monitor_2.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
monitor = np.concatenate([monitor1, monitor2])

# monitor = np.loadtxt("plots/mt10/mt10_monitor.csv", delimiter=',', skiprows=2, max_rows=number_rows)
# print(monitor[1])
print("len monitor:",len(monitor))

titles = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2','button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2','Average']
max_sucess_rates = []

# Get Task max success rate:
for i in range(10):
    filter_tasks = monitor[:, 3] == i
    print(i)
    # time steps:
    x_time_steps_values = monitor[:, 1][filter_tasks]
    print(len(x_time_steps_values))
    size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
    print(size)
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

    max_sucess_rates.append(max(y_success_values))

#
avg = np.average(max_sucess_rates)
max_sucess_rates.append(avg)

max_sucess_rates = [item * 100 for item in max_sucess_rates]

# create Figure
fig, ax = plt.subplots()
y_pos = np.arange(len(titles))
print(y_pos)
print(max_sucess_rates)
ax.barh(y_pos, max_sucess_rates)
ax.set_yticks(y_pos, labels=titles)
ax.set_xlabel('Maximum Success Rate (%)')
ax.set_ylabel("Environment")
plt.tight_layout()

fig.set_size_inches(11, 6)
plt.savefig("plots/finished_pdf_plots/mt10_all_tasks_max_success_rate.pdf")
plt.show()
