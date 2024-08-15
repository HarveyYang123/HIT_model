

import matplotlib.pyplot as plt

# 设置matplotlib使用TrueType字体
plt.rcParams['pdf.fonttype'] = 42  # 这对于PDF输出很重要
plt.rcParams['ps.fonttype'] = 42   # 这对于PostScript输出很重要

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['axes.labelsize'] = 30

fig, axs = plt.subplots(1, 1, figsize=(12, 4))

names = ['HIT', 'Vanilla', 'Late', 'Early']
data = {
    'HIT-AverageResponseTime': [0.9643],
    'Vanilla Two-tower-AverageResponseTime': [0.926],
    'Late Interaction-AverageResponseTime': [17.86],
    'Early Interaction-AverageResponseTime': [1.747]
}


values = [val[0] for val in data.values()]

# 设置柱子之间的间距
spacing = 0.1

# 计算调整后的x轴位置
x_positions = [i + spacing * i for i in range(len(names))]

axs.bar(names, values, width=0.5)
axs.set_ylabel('millisecond')
# axs.set_title('QPS=35000 Average Response Time')
axs.set_xticks(range(len(names)))
axs.set_xticklabels(names)



# 调整子图布局
plt.tight_layout()

plt.savefig('../figure/efficiency-1.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('../figure/efficiency-1.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.savefig("../figure/efficiency-1.pdf",format="pdf")
plt.show()
