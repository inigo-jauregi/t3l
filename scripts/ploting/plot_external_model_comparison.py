import matplotlib.pyplot as plt


x = [1, 10, 100]
y_mbart = [49.09, 60.34, 74.44]
y_mbart_jttl = [42.23, 58.77, 70.94]
y_mbert = [39.61, 43.19, 67.49]
y_xlmr = [71.07, 79.3, 81.37]

plt.plot(x, y_mbert, 'g', marker="v", markersize=8, label="mBERT")
plt.plot(x, y_mbart, 'k', marker="+", markersize=8, label="mBART")
plt.plot(x, y_xlmr, 'r', marker="x", markersize=8, label="XLM-R")
plt.plot(x, y_mbart_jttl, 'b', marker="|", markersize=8, label="mBART-T3L")
plt.xscale('log')
plt.xlim([0.8, 110])
plt.ylabel('mRP (%)', fontsize=12)
plt.xlabel('# of tuning samples', fontsize=12)
plt.xticks([1, 10, 100],  labels=['0', '10', '100'])
plt.legend(loc="lower right")

plt.show()
