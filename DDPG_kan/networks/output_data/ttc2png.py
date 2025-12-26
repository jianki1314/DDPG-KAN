import os
import subprocess

def convert_xml_to_csv(folder_path):
    # 获取所有 .xml 文件的路径
    xml_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.xml')]
    
    # 确保输出文件路径存在
    output_folder = os.path.join(folder_path, '../ssm_png')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'combined_ttc_plot.png')  # 合并后的图形文件名
    csv_output_file = os.path.join(output_folder, 'combined_ttc_data.csv')  # 输出CSV文件路径
    
    # 构建并执行命令，包含所有文件和参数
    command = (
        f"python /home/jian/sumo/tools/visualization/plotXMLAttributes.py "
        f"{' '.join(xml_files)} "  # 将所有 xml 文件路径合并成一个字符串
        f"-x time --xlabel 'Time [s]' "
        f"-y value --ylabel 'TTC [s]' "
        f"--title 'time to collision over simulation time for all vehicles' "
        f"--scatterplot "
        f"-o {output_file} "
        f"--csv-output {csv_output_file}"  # 添加CSV输出参数
    )
    
    # 执行命令并捕获输出
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    print(f"Processed all files and generated combined plot.")

    # 解析输出以获取点数
    with open(csv_output_file, 'r') as file:
        data_points = file.readlines()
        points_count = len(data_points) - 1  # 减去标题行
        print(f"Total number of data points in the output plot: {points_count}")

# 替换为您的文件夹路径
folder_path = '../networks/output_data/output_td3/ssm'
convert_xml_to_csv(folder_path)
