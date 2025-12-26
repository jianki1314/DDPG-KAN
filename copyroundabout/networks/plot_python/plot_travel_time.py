import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 指定两个文件夹路径和要匹配的文件模式
folders = {
    'Krauss': '/home/jian/sumo/DDPG_kan/networks/output_data/output_default/tripinfo',
    'PPO': '/home/jian/sumo/copyroundabout/networks/output_data/output_ppo/tripinfo'
}
file_pattern = '*-tripinfo-output.csv'

all_data = []

for model, file_path in folders.items():
    all_files = glob.glob(os.path.join(file_path, file_pattern))
    
    for filename in all_files:
        # 首先检查文件是否为空
        if os.stat(filename).st_size > 0:
            try:
                df = pd.read_csv(filename, sep=';')
                # 判断DataFrame是否为空
                if not df.empty:
                    # 添加一个表示模型的列
                    df['Model'] = model
                    all_data.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping invalid file: {filename}")
        else:
            print(f"Skipping empty file: {filename}")

# 确保all_data非空
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 筛选rl_0车辆的数据行 **注意** 如果tripinfo_id一列名发生变化，请相应调整。
    rl_0_df = combined_df.loc[combined_df['tripinfo_id'] == 'rl_0'].copy()
    model_order = list(folders.keys())
    rl_0_df.loc[: , 'Model'] = pd.Categorical(rl_0_df['Model'], categories=model_order, ordered=True)
    model_stats = rl_0_df.groupby('Model')['tripinfo_duration'].agg(['median', 'mean']).reset_index()
    
    # 打印结果
    print("Statistics of travel time (s) for each model:")
    print(model_stats)

    # 计算和比较平均值的相差百分比
    default_mean = model_stats.loc[model_stats['Model'] == 'Krauss', 'mean'].values[0]
    ddpg_kan_mean = model_stats.loc[model_stats['Model'] == 'PPO', 'mean'].values[0]
    # ddpg_kan_mean = model_stats.loc[model_stats['Model'] == 'TD3-kan', 'mean'].values[0]
    if default_mean > ddpg_kan_mean:
        percent_difference = ((default_mean - ddpg_kan_mean) / default_mean) * 100
        print(f"下降了 {percent_difference:.4f}%")
    else:
        percent_difference = ((ddpg_kan_mean - default_mean) / default_mean) * 100
        print(f"上升了 {percent_difference:.4f}%")
    
    plt.figure(figsize=(10, 8))
    
    # 使用Seaborn绘制箱线图，其中hue参数用于区分不同模型生成的数据点
    sns.boxplot(data=rl_0_df, x='Model', y='tripinfo_duration', order=model_order)
    
    # 调整标签和标题以反映实际内容
    plt.xlabel('Model')
    plt.ylabel('travel time (s)')
    plt.title('travel time for rl_0 Across Models')
    
    plt.show()
else:
    print("No files found matching the pattern.")


'''
# 指定文件夹路径和文件模式
file_path = '/home/jian/sumo/copyroundabout/networks/output_data/output_folder'  # 你的CSV文件夹路径
file_pattern = '*-tripinfo-output.csv'  # 根据您的文件名进行调整

# 读取所有匹配的CSV文件
all_files = glob.glob(os.path.join(file_path, file_pattern))
all_data = []

for filename in all_files:
    df = pd.read_csv(filename, sep=';')
    all_data.append(df)

# 单独读取rl_0的车辆数据
if all_data:
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    # 筛选tripinfo_id为rl_0的数据行
    rl_0_df = combined_df.loc[combined_df['tripinfo_id'] == 'rl_0']

    # 取出rl_0的电耗量
    emissions_electricity_abs = rl_0_df['emissions_electricity_abs']

    # 计算总体平均值和中位数
    overall_average = emissions_electricity_abs.mean()
    overall_median = emissions_electricity_abs.median()

    print(f"Overall average Et for rl_0: {overall_average} wh")
    print(f"Overall median Et for rl_0: {overall_median} wh")

    # 绘制箱线图
    plt.figure(figsize=(10, 8))
    sns.boxplot(y=emissions_electricity_abs)
    plt.xlabel('Model')
    plt.ylabel('Emissions Electricity Abs (wh)')
    plt.title('Emissions Electricity Boxplot for rl_0')
    plt.show()
else:
    print("No files found matching the pattern.")
'''




'''
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 指定文件夹路径和文件模式
file_path = '/home/jian/sumo/copyroundabout/networks/output_data/output_folder'  # 你的CSV文件夹路径
file_pattern = '*-tripinfo-output.csv'  # 根据您的文件名进行调整

# 读取所有匹配的CSV文件
all_files = glob.glob(os.path.join(file_path, file_pattern))
all_data = []

for filename in all_files:
    df = pd.read_csv(filename, sep=';')
    all_data.append(df)

# 单独读取rl_0的车辆数据
if all_data:
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    # 筛选tripinfo_id为rl_0的数据行
    rl_0_df = combined_df.loc[combined_df['tripinfo_id'] == 'rl_0']

    # 取出rl_0的电耗量
    emissions_electricity_abs = rl_0_df['tripinfo_duration']

    # 计算总体平均值和中位数
    overall_average = emissions_electricity_abs.mean()
    overall_median = emissions_electricity_abs.median()

    print(f"Overall average duration for rl_0: {overall_average} s")
    print(f"Overall median duration for rl_0: {overall_median} s")

    # 绘制箱线图
    plt.figure(figsize=(10, 8))
    sns.boxplot(y=emissions_electricity_abs)
    plt.xlabel('Model')
    plt.ylabel('tripinfo_duration (s)')
    plt.title('travel_time for rl_0')
    plt.show()
else:
    print("No files found matching the pattern.")
'''