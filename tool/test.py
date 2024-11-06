import matplotlib.pyplot as plt

# 数据
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

# 创建柱状图
plt.bar(categories, values)

# 在柱状图上方添加数值
for i in range(len(categories)):
    plt.text(x=i, y=values[i]+1, s=values[i], ha='center', fontsize=10)

# 设置图表标题和坐标轴标签
plt.title('柱状图示例')
plt.xlabel('类别')
plt.ylabel('值')

# 显示图表
plt.show()