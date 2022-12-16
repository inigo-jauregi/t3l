import matplotlib.pyplot as plt


x = [2.92, 4.97, 6.91, 7.57, 14.21, 14.69, 22.44, 23.21, 25.37, 26.97, 31.75, 37.41, 42.77, 44.96,
     47.59, 51.65, 54.98, 58.22, 60.59, 60.62, 60.85]
y_zero_shot = [39.02, 38.12, 39.92, 65.7, 62.98, 66.5, 73.03, 78.17, 78.01, 77.97, 78.19, 78.37,
               77.97, 77.75, 78.05, 78.51, 78.23, 77.61, 75.65, 75.75, 76.31]
y_few_shot_10 = [35.01, 35.19, 46.71, 64.78, 62.88, 66.93, 73.52, 78.05, 78.05, 77.68, 78.09, 77.85,
                 77.95, 77.81, 78.21, 78.09, 78.03, 77.87, 75.88, 75.98, 76.42]
y_few_shot_100 = [55.28, 53.76, 50.28, 61.98, 68.03, 69.15, 73.9, 79.13, 78.61, 78.49, 78.45, 79.05,
                  78.71, 78.23, 78.87, 78.65, 78.81, 78.41, 76.3, 76.28, 76.86]
x_baseline = [59.85]
y_baseline = [71.85]
y_baseline_few_shot_100 = [71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85,
                           71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85, 71.85]

plt.plot(x, y_zero_shot, 'g', marker="v", markersize=8, label="T3L (ZS)")
plt.plot(x, y_few_shot_10, 'b', marker="+", markersize=8, label="T3L (FS-10)")
plt.plot(x, y_few_shot_100, 'r', marker="x", markersize=8, label="T3L (FS-100)")
plt.plot(x_baseline, y_baseline, 'o', markersize=8, markeredgecolor='k', markerfacecolor='k', label="LM (FS-100)")
plt.xlim([12, 61])
plt.ylim([64, 80])
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('Test BLEU', fontsize=12)
plt.xticks([15, 25, 35, 45, 55],  labels=['15', '25', '35', '45', '55'])
plt.legend(loc="lower right")

plt.show()
