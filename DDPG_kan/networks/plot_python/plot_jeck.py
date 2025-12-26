import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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

# 读取DDPG-kan的数据
ddpg_kan_files = glob.glob(os.path.join(folders['AC-KAN'], file_pattern))
ddpg_kan_data = []

for filename in ddpg_kan_files:
    df = pd.read_csv(filename)
    # 从文件名中提取回合数，添加到数据框中
    episode_number = int(os.path.basename(filename).split('-')[0])
    df['episode'] = episode_number
    ddpg_kan_data.append(df)

# 合并所有DDPG-kan的数据
ddpg_kan_combined_df = pd.concat(ddpg_kan_data, ignore_index=True)

# 识别DDPG-kan中的异常数据
zero_acc_duration_ddpg_kan = ddpg_kan_combined_df['agent_jeck'].rolling(window=250, center=True).apply(lambda x: (x == 0).all())
ddpg_kan_combined_df['anomaly'] = zero_acc_duration_ddpg_kan
ddpg_kan_combined_df = ddpg_kan_combined_df[ddpg_kan_combined_df['anomaly'] == False]

# 计算DDPG-kan每回合的jeck标准差平均值
ddpg_kan_std_mean = ddpg_kan_combined_df.groupby('episode')['agent_jeck'].std().mean()
print(f"AC-KAN模型每回合jeck标准差的平均值: {ddpg_kan_std_mean:.4f}")

# 遍历除DDPG-kan以外的模型
for folder_name, folder_path in folders.items():
    if folder_name == 'AC-KAN':
        continue  # 跳过DDPG-kan模型的自身对比

    print(f"Processing folder: {folder_name}")
    all_files = glob.glob(os.path.join(folder_path, file_pattern))

    all_data = []
    for filename in all_files:
        df = pd.read_csv(filename)
        # 从文件名中提取回合数，添加到数据框中
        episode_number = int(os.path.basename(filename).split('-')[0])
        df['episode'] = episode_number
        all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 识别异常数据
    zero_acc_duration = combined_df['agent_jeck'].rolling(window=250, center=True).apply(lambda x: (x == 0).all())
    combined_df['anomaly'] = zero_acc_duration
    combined_df = combined_df[combined_df['anomaly'] == False]

    # 计算每回合的jeck标准差平均值
    std_mean = combined_df.groupby('episode')['agent_jeck'].std().mean()
    print(f"{folder_name} 模型每回合jeck标准差的平均值: {std_mean:.4f}")

    # 绘制该模型与DDPG-kan的对比柱状图
    plt.figure(figsize=(10, 8))

    # 绘制DDPG-kan的加速度柱状图（浅蓝色）
    plt.hist(ddpg_kan_combined_df['agent_jeck'], bins=50, alpha=0.6, label='AC-KAN', color='blue', edgecolor='black', histtype='stepfilled')

    # 绘制当前模型的加速度柱状图（橙色）
    plt.hist(combined_df['agent_jeck'], bins=50, alpha=0.6, label=folder_name, color='orange', edgecolor='black', histtype='stepfilled')

    # 设置图形标题和图例
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']  # 设置为支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    log_dir = "../picture"
    plt.xlabel(r"Driving Comfort $(\mathrm{m/s^3})$", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.axhline(0, color='black', linewidth=1)  # 以0为中心坐标轴
    plt.xlim(-17.5, 17.5)
    plt.grid(True)
    plt.legend()

    # 保存图形
    plt.savefig(os.path.join(log_dir, f"{folder_name}_vs_AC-KAN_jeck.png"), dpi=300)
    plt.show()
    
