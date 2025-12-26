import os
import subprocess

def convert_xml_to_csv(folder_path):
    file_count = 0  # 初始化计数器
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            xml_file = os.path.join(folder_path, filename)
            # 构建并执行命令
            command = f"python /home/jian/sumo/tools/xml/xml2csv.py {xml_file}"
            try:
                result = subprocess.run(command, shell=True, check=True)
                print(f"Processed {filename}")
                file_count += 1  # 成功处理后递增计数器
                
                # 可选：删除原始的xml文件
                os.remove(xml_file)
                print(f"Deleted {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")
    
    print(f"一共转换的文件数量： {folder_path}: {file_count}")  # 输出总数

# 替换为您的文件夹路径
folder_path = '../networks/output_data/output_ddpg/tripinfo'
folder_path1 = '../networks/output_data/output_ppo/tripinfo'
folder_path2 = '../networks/output_data/output_td3/tripinfo'
folder_path3 = '../networks/output_data/output_sac/tripinfo'
folder_path4 = '../networks/output_data/output_krauss/tripinfo'
convert_xml_to_csv(folder_path)

