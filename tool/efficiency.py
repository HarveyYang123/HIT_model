

import matplotlib.pyplot as plt

# 设置matplotlib使用TrueType字体
plt.rcParams['pdf.fonttype'] = 42  # 这对于PDF输出很重要
plt.rcParams['ps.fonttype'] = 42   # 这对于PostScript输出很重要


# 假设这是你的数据
data = {
    'QPS': [35000, 52500, 53500, 54000, 55000, 57000, 61500],
    'HIT-AverageResponseTime': [0.9643, 1.063, 1.103, 1.115, 1.162, 1.238, 1.526],
    'HIT-SuccessRate': [0.9999, 0.9998, 0.9992, 0.999, 0.999, 0.9982, 0.9135],

    'Vanilla Two-tower-AverageResponseTime': [0.926, 0.9818],
    'Vanilla Two-tower-SuccessRate': [0.9999, 0.9997],

    'Late Interaction-AverageResponseTime': [17.86],
    'Late Interaction-SuccessRate': [0.1878],

    'Early Interaction-AverageResponseTime': [1.747, 3.156],
    'Early Interaction-SuccessRate': [0.9994, 0.8123]
}

QPS=35000,耗时由17.86毫秒降到 0.9643毫秒

data_2 = {
    'HIT-SuccessRate': [0.9999],
    'Vanilla Two-tower-SuccessRate': [0.9999],
    'Late Interaction-SuccessRate': [0.1878],
    'Early Interaction-SuccessRate': [0.9994]
}


# HIT：https://adt-test.woa.com/#/reports/cupaiZhaohuiInfer/b-5259739900c84885adccce2ccc88782b
# Vanilla Two-tower:https://adt-test.woa.com/#/reports/cupaiZhaohuiInfer/b-eea2e095bab54464b7683e06a16907c6
# MVKE Late Interaction: https://adt-test.woa.com/#/reports/cupaiZhaohuiInfer/b-b0b8fa773de74540ae34a0eb7826b4ea
# Early Interaction: https://adt-test.woa.com/#/reports/cupaiZhaohuiInfer/b-8e78d2489bc942ec82f89b4b91141a43
# 创建一个大图和子图
fig, axs = plt.subplots(2, 2, figsize=(12, 4))

names = ['HIT', 'Vanilla', 'Late', 'Early']
data_1 = {
    'HIT-AverageResponseTime': [0.9643],
    'Vanilla Two-tower-AverageResponseTime': [0.926],
    'Late Interaction-AverageResponseTime': [17.86],
    'Early Interaction-AverageResponseTime': [1.747]
}


values_1 = [val[0] for val in data_1.values()]

# Create the bar chart
axs[0, 0].bar(names, values_1)

# Set labels and title
axs[0, 0].set_ylabel('millisecond')
axs[0, 0].set_title('QPS=35000 Average Response Time')

# Set the x-ticks and x-tick labels
axs[0, 0].set_xticks(range(len(names)))
axs[0, 0].set_xticklabels(names, rotation=45)




data_3 = {
    'HIT-SuccessRate': [0.9999],
    'Vanilla Two-tower-SuccessRate': [0.9999],
    'Late Interaction-SuccessRate': [0.1878],
    'Early Interaction-SuccessRate': [0.9994]
}



values_2 = [val[0] for val in data_3.values()]

# Create the bar chart
axs[0, 1].bar(names, values_2)

# Set labels and title
axs[0, 1].set_ylabel('Success Rate')
axs[0, 1].set_title('QPS=35000 Response Success Rate')

# Set the x-ticks and x-tick labels
axs[0, 1].set_xticks(range(len(names)))
axs[0, 1].set_xticklabels(names, rotation=45)



names_1 = ['HIT', 'Vanilla', 'Early']
data_3 = {
    'HIT-AverageResponseTime': [1.063],
    'Vanilla Two-tower-AverageResponseTime': [0.9818],
    'Early Interaction-AverageResponseTime': [3.156]
}


values_3 = [val[0] for val in data_3.values()]

# Create the bar chart
axs[1, 0].bar(names_1, values_3)

# Set labels and title
axs[1, 0].set_ylabel('millisecond')
axs[1, 0].set_title('QPS=52500 Average Response Time')

# Set the x-ticks and x-tick labels
axs[1, 0].set_xticks(range(len(names_1)))
axs[1, 0].set_xticklabels(names_1, rotation=45)




data_4 = {
    'HIT-SuccessRate': [0.9998],
    'Vanilla Two-tower-SuccessRate': [0.9997],
    'Early Interaction-SuccessRate': [0.8123]
}



values_4 = [val[0] for val in data_4.values()]

# Create the bar chart
axs[1, 1].bar(names_1, values_4)

# Set labels and title
axs[1, 1].set_ylabel('Success Rate')
axs[1, 1].set_title('QPS=52500 Response Success Rate')

# Set the x-ticks and x-tick labels
axs[1, 1].set_xticks(range(len(names_1)))
axs[1, 1].set_xticklabels(names_1, rotation=45)

# 调整子图布局
plt.tight_layout()

plt.savefig('../figure/efficiency.eps', dpi=300)  # eps文件，用于LaTeX
plt.savefig('../figure/efficiency.svg', dpi=300)  # svg文件，可伸缩矢量图形 (Scalable Vector Graphics)
plt.savefig("../figure/efficiency.pdf",format="pdf")
plt.show()
