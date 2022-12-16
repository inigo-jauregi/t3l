import matplotlib.pyplot as plt


x = [0.32, 1.01, 2.95, 7.39, 11.77, 17.68, 22.49, 24.4, 28.54, 30.69, 34.09, 35.64, 36.01, 37.64, 38.41]
y_zero_shot = [33.39, 37.09, 39.68, 47.9, 52.63, 56.68, 58.97, 59.53, 62.09, 64, 67.07, 67.69, 67.47, 68.57, 69.47]
y_few_shot_10 = [33.26, 37.13, 40.08, 49.03, 53.24, 56.81, 60.01, 60.25, 64.16, 64.36, 66.99, 68.07, 68.69,
                 69.23, 69.51]
y_few_shot_100 = [34.31, 37.45, 40.6, 47.79, 55.12, 57.97, 59.69, 63.96, 63.98, 66.89, 70.11, 71.39, 71.95, 72.6, 72.26]
x_baseline = [10.5]
y_baseline = [36.26]
x_baseline_mt = [38.41]
y_baseline_mt = [61.93]
y_baseline_few_shot_100 = [36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26, 36.26,
                           36.26, 36.26]

plt.plot(x, y_zero_shot, 'g', marker="v", markersize=8, label="T3L (ZS)")
plt.plot(x, y_few_shot_10, 'b', marker="+", markersize=8, label="T3L (FS-10)")
plt.plot(x, y_few_shot_100, 'r', marker="x", markersize=8, label="T3L (FS-100)")
# plt.plot(x, y_baseline_few_shot_100, 'k--', label="LM (FS-100)")
plt.plot(x_baseline, y_baseline, 'o', markersize=8, markeredgecolor='k', markerfacecolor='k', label="LM (FS-100)")
plt.plot(x_baseline_mt, y_baseline_mt, 'h', markersize=8, markeredgecolor='brown', markerfacecolor='brown',
         label="LM_mt (FS-100)")
plt.xlim([10, 40])
plt.ylim([30, 75])
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('Test BLEU', fontsize=12)
plt.xticks([15, 20, 25, 30, 35],  labels=['15', '20', '25', '30', '35'])
plt.legend(loc="upper left")

plt.show()
