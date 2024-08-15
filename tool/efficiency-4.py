

import matplotlib.pyplot as plt

# 设置matplotlib使用TrueType字体
plt.rcParams['pdf.fonttype'] = 42  # 这对于PDF输出很重要
plt.rcParams['ps.fonttype'] = 42   # 这对于PostScript输出很重要

plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['axes.labelsize'] = 15


fig, axs = plt.subplots(1, 1, figsize=(6, 4))

names = ['HIT', 'Vanilla', 'Early']
data = {
    'HIT-SuccessRate': [0.99],
    'Vanilla Two-tower-SuccessRate': [0.99],
    'Early Interaction-SuccessRate': [0.81]
}


values = [val[0] for val in data.values()]

for i in range(len(names)):
    axs.text(x=i, y=values[i], s=values[i], ha='center', fontsize=20)


axs.bar(names, values, width=0.5)
axs.set_ylabel('Success Rate', fontsize=25)
# axs.set_title('QPS=52500 Response Success Rate')
axs.set_xticks(range(len(names)))
axs.set_xticklabels(names)



# 调整子图布局
plt.tight_layout()

plt.savefig('../figure/efficiency-4.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('../figure/efficiency-4.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.savefig("../figure/efficiency-4.pdf",format="pdf")
plt.show()
