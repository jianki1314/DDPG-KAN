import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 指定文件夹路径和匹配文件模式
folders = {
    'Krauss': '/home/jian/sumo/copyroundabout/networks/output_data/output_krauss/tripinfo',
    'EIDM': '../networks/output_data/output_eidm/tripinfo',
    'SAC': '/home/jian/sumo/copyroundabout/networks/output_data/output_sac/tripinfo',
    'TD3': '/home/jian/sumo/copyroundabout/networks/output_data/output_td3/tripinfo',
    'PPO': '/home/jian/sumo/copyroundabout/networks/output_data/output_ppo/tripinfo',
    'DDPG': '/home/jian/sumo/copyroundabout/networks/output_data/output_ddpg/tripinfo',
    'AC-KAN': '../networks/output_data/output_kan/tripinfo',
    'TD3-KAN': '../networks/output_data/output_td3/tripinfo'
}
file_pattern = '*-tripinfo-output.csv'

all_data = []

# 遍历文件夹并读取数据
for model, file_path in folders.items():
    all_files = glob.glob(os.path.join(file_path, file_pattern))
    
    for filename in all_files:
        # 检查文件是否为空
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
    
    # 筛选rl_0车辆的数据行
    rl_0_df = combined_df.loc[combined_df['tripinfo_id'] == 'rl_0']
    
    # 按照模型计算每个模型的中位数和平均值
    model_stats = rl_0_df.groupby('Model')['emissions_electricity_abs'].agg(['median', 'mean']).reset_index()

    # 按照原始顺序对模型排序
    model_stats['Model'] = pd.Categorical(model_stats['Model'], categories=folders.keys(), ordered=True)
    model_stats = model_stats.sort_values('Model')

    # 打印结果
    print("Statistics of Emissions Electricity Abs (wh) for each model:")
    print(model_stats)

    # 与Krauss模型进行比较
    krauss_mean = model_stats.loc[model_stats['Model'] == 'Krauss', 'mean'].values[0]
    print(f"Krauss model mean emissions: {krauss_mean:.4f} wh")

    for model in model_stats['Model'].unique():
        if model != 'Krauss':
            model_mean = model_stats.loc[model_stats['Model'] == model, 'mean'].values[0]
            if krauss_mean > model_mean:
                percent_difference = ((krauss_mean - model_mean) / krauss_mean) * 100
                print(f"{model} 电耗下降了 {percent_difference:.4f}%")
            else:
                percent_difference = ((model_mean - krauss_mean) / krauss_mean) * 100
                print(f"{model} 电耗上升了 {percent_difference:.4f}%")
    
    plt.figure(figsize=(10, 8))

    # 定义调色板，将DDPG-kan模型设为橙色，其他模型保持默认颜色
    palette = {model: 'C0' for model in folders.keys()}  # 默认颜色
    palette['AC-KAN'] = 'orange'  # 将DDPG-kan设为橙色

    # 按照模型顺序绘制箱线图，设置`hue`为`Model`并关闭图例
    sns.boxplot(data=rl_0_df, x='Model', y='emissions_electricity_abs', order=folders.keys(), palette=palette, hue='Model', legend=False)

    # 调整标签和标题
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei']  # 设置为支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    log_dir = "../picture"
    plt.xlabel('Model', fontsize=15)  # 设置字体大小
    plt.ylabel(' Energy Consumption/wh', fontsize=15)  # 设置字体大小
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "Et.png"), dpi=300)
    plt.show()
else:
    print("No files found matching the pattern.")
