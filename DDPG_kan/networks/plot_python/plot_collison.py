import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

# 指定文件夹路径和匹配文件模式
folders = {
    'Krauss': '../networks/output_data/output_default/state',
    'EIDM': '../networks/output_data/output_eidm/state',
    'SAC': '/home/jian/sumo/copyroundabout/networks/output_data/output_sac/state',
    'TD3': '/home/jian/sumo/copyroundabout/networks/output_data/output_td3/state',
    'PPO': '/home/jian/sumo/copyroundabout/networks/output_data/output_ppo/state',
    'DDPG': '/home/jian/sumo/copyroundabout/networks/output_data/output_ddpg/state',
    'LSTM': '../networks/output_data/output_lstm/state',
    'AC-KAN': '../networks/output_data/output_kan/state'
}
file_pattern = '*-state.csv'

# 初始化存储碰撞次数的字典
collision_counts = {model: [] for model in folders.keys()}

# 遍历每个文件夹
for folder_name, folder_path in folders.items():
    print(f"Processing folder: {folder_name}")
    all_files = glob.glob(os.path.join(folder_path, file_pattern))

    for filename in all_files:
        df = pd.read_csv(filename)
        
        if not df.empty:
            # 获取每个文件中agent_collision的最后一个值
            last_collision_count = df['agent_collision'].iloc[-1]
            # 存储每个模型的碰撞次数
            collision_counts[folder_name].append(last_collision_count)
        else:
            print(f"File {filename} is empty and will be skipped.")

# 计算每个模型碰撞次数的平均值
average_collisions = {model: sum(counts) / len(counts) if counts else 0 for model, counts in collision_counts.items()}
print(average_collisions)

# 创建柱状图
plt.figure(figsize=(12, 8))
colors = ['orange' if model == 'AC-KAN' else 'blue' for model in average_collisions.keys()]
plt.bar(average_collisions.keys(), average_collisions.values(), color=colors, alpha=0.7)

# 设置图形标题和标签
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']  # 设置为支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.xlabel("模型", fontsize=15)
plt.ylabel("碰撞预警平均值", fontsize=15)
# plt.title("各算法模型的碰撞预警平均值对比", fontsize=18)
plt.xticks(rotation=45)
plt.grid(axis='y')

# 保存图形
log_dir = "../picture"
os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在则创建
plt.savefig(os.path.join(log_dir, "average_collision_counts.png"), dpi=300)
plt.show()
