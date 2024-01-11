import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
# seaborn.set(font_scale=1.2)
plt.style.use("plots/subfigure.mplstyle")

# number_rows = 500000
values_combined = 100
monitor = np.loadtxt("plots/mt50/mt50_monitor.csv", delimiter=',', skiprows=2,)# max_rows=number_rows)
# print(monitor[1])
print("len monitor:",len(monitor))

titles = ["assembly-v2","basketball-v2","bin-picking-v2", "box-close-v2", "button-press-topdown-v2", "button-press-topdown-wall-v2", "button-press-v2", "button-press-wall-v2", "coffee-button-v2", "coffee-pull-v2", "coffee-push-v2", "dial-turn-v2", "disassemble-v2", "door-close-v2", "door-lock-v2", "door-open-v2", "door-unlock-v2", "hand-insert-v2", "drawer-close-v2", "drawer-open-v2", "faucet-open-v2", "faucet-close-v2", "hammer-v2", "handle-press-side-v2", "handle-press-v2","handle-pull-side-v2", "handle-pull-v2", "lever-pull-v2", "peg-insert-side-v2", "pick-place-wall-v2", "pick-out-of-hole-v2", "reach-v2", "push-back-v2", "push-v2", "pick-place-v2", "plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2", "plate-slide-back-side-v2", "peg-unplug-side-v2","soccer-v2", "stick-push-v2","stick-pull-v2", "push-wall-v2", "reach-wall-v2", "shelf-place-v2","sweep-into-v2","sweep-v2", "window-open-v2", "window-close-v2","Average"]
max_sucess_rates = []

# Get Task max success rate:
for i in range(50):
    filter_tasks = monitor[:, 3] == i
    print(i)
    # time steps:
    x_time_steps_values = monitor[:, 1][filter_tasks]
    size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
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
print(len(y_pos), len(max_sucess_rates))
ax.barh(y_pos, max_sucess_rates,)
ax.set_yticks(y_pos, labels=titles)
ax.set_xlabel('Maximum Success Rate (%)')
ax.set_ylabel("Environment")

fig.set_size_inches(8, 10)
plt.tight_layout()
plt.savefig("plots/finished_pdf_plots/mt50_all_tasks_max_success_rate.pdf")
plt.show()
