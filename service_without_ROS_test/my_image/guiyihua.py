import numpy as np

# 读取深度图像数据
data = np.load('/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test/my_image/my_dep_2.npy')

# 计算数据的最小值和最大值
min_value = np.min(data)
max_value = np.max(data)

# 将数据归一化到 [0, 1] 的范围内
data_normalized = (data - min_value) / (max_value - min_value)

# 将归一化后的数据保存为 .npy 文件
np.save('/home/chimy/projects/biyesheji/cloth-segmentation/service_without_ROS_test/my_image/my_dep_2_normal.npy', data_normalized)
