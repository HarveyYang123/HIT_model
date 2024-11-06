

import matplotlib.pyplot as plt

# 设置matplotlib使用TrueType字体
plt.rcParams['pdf.fonttype'] = 42  # 这对于PDF输出很重要
plt.rcParams['ps.fonttype'] = 42   # 这对于PostScript输出很重要

plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['axes.labelsize'] = 15

fig, axs = plt.subplots(1, 1, figsize=(6, 4))

names = ['HIT', 'Vanilla', 'Early', 'Late']
data = {
    'HIT-AverageResponseTime': [0.96],
    'Vanilla Two-tower-AverageResponseTime': [0.92],
    'Early Interaction-AverageResponseTime': [1.75],
    'Late Interaction-AverageResponseTime': [17.86]
}


values = [val[0] for val in data.values()]
axs.bar(names, values, width=0.5)

# for i in range(len(names)):
#     axs.text(x=i, y=values[i], s=values[i], ha='center', fontsize=20)

axs.set_ylabel('millisecond', fontsize=25)
# axs.set_title('QPS=35000 Average Response Time')
axs.set_xticks(range(len(names)))
axs.set_xticklabels(names)
# axs.grid(True)

# 调整子图布局
plt.tight_layout()

plt.savefig('../figure/efficiency-1.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('../figure/efficiency-1.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.savefig("../figure/efficiency-1.pdf",format="pdf")
plt.show()
