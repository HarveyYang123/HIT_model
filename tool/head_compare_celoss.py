

import matplotlib.pyplot as plt

# 设置matplotlib使用TrueType字体
plt.rcParams['pdf.fonttype'] = 42  # 这对于PDF输出很重要
plt.rcParams['ps.fonttype'] = 42   # 这对于PostScript输出很重要


plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 15
# 假设这是你的数据
data = {
    'head': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'Alibaba_AUC': [0.72, 0.7226, 0.7202, 0.7199, 0.7179, 0.7161, 0.7176, 0.718, 0.716, 0.7175, 0.72, 0.7185, 0.7144, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'Alibaba_Logloss': [0.2233,0.2104,0.2242,0.2342,0.2479,0.2445,0.2584,0.2653,0.267,0.2686,0.265,0.2653,0.2659,31.6041,31.6041,31.6041,31.6041,31.6041,31.6041,31.6041,31.6041, 31.6041, 31.6041, 31.6041],
    'MovieLens_AUC': [0.8984,0.9012,0.8979,0.8984,0.8967,0.8975,0.8817,0.8141,0.8979,0.8983,0.7882,0.8981,0.8974,0.7879,0.9011,0.8959,0.8678,0.8279,0.8575,0.7533,0.5,0.5,0.5,0.5],
    'MovieLens_Logloss': [0.3143,0.3107,0.3129,0.3119,0.32,0.3157,0.4047,0.4281,0.3157,0.3154,0.4318,0.3126,0.3148,0.4329,0.3117,0.316,0.5955,0.5539,0.6545,0.8733,7.6549,7.6549,7.6549,7.6549],
    'Amazon_AUC': [0.871,0.8711,0.8711,0.8721,0.8766,0.872,0.8717,0.815,0.8706,0.871,0.8694,0.6977,0.8715,0.5,0.8704,0.8601,0.8717,0.5645,0.5,0.8152,0.5457],
    'Amazon_Logloss': [0.3313,0.3321,0.3315,0.3291,0.3247,0.3328,0.3332,0.3785,0.3307,0.3308,0.335,4.0882,0.3299,6.6922,0.3419,0.3629,0.8601,6.011,6.6922,0.3765,6.2162]
}

fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))


ax2.plot(data['head'][:6], data['Alibaba_Logloss'][:6], 'o-', label='Alibaba')
ax2.plot(data['head'][:6], data['MovieLens_Logloss'][:6], '^-', label='MovieLens')
ax2.plot(data['head'][:6], data['Amazon_Logloss'][:6], 'p-', label='Amazon')
ax2.set_xlabel('# heads.', fontsize=30)
ax2.set_xlim(1, 6, 1)
ax2.set_ylim(0.2, 0.35, 0.01)
# ax2.set_ylabel('CEloss')
ax2.grid(True)
ax2.legend(loc='center left', prop={'family': 'Times New Roman', 'size': 22})

# 调整子图布局
plt.tight_layout()

plt.savefig('../figure/head_num_celoss_v2.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('../figure/head_num_celoss_v2.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.savefig("../figure/head_num_celoss_v2.pdf",format="pdf")
plt.show()
