import torch
import numpy as np
import h5py
import os
from kan import KAN

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 加载数据
def load_data(folder_path):
    all_states = []
    all_actions = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            with h5py.File(file_path, 'r') as f:
                states = np.array(f['states'])
                actions = np.array(f['actions'])
                
                # 确保动作数据是二维的
                if actions.ndim == 1:
                    actions = actions[:, np.newaxis]
                
                all_states.append(states)
                all_actions.append(actions)
    
    # 合并所有数据
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)
    
    return all_states, all_actions

# 文件夹路径
folder_path = 'hdf5'
states, actions = load_data(folder_path)

# 转换为PyTorch张量
states = torch.from_numpy(states).float().to(device)
actions = torch.from_numpy(actions).float().to(device)

# 创建数据集字典
dataset = {
    'train_input': states,
    'train_label': actions,
    'test_input': states,  # 使用训练数据作为测试数据
    'test_label': actions  # 使用训练数据作为测试数据
}

# 初始化KAN模型
input_dim = states.shape[1]  # 状态空间的维度
output_dim = actions.shape[1]  # 动作空间的维度
print("Input dimensions:", input_dim)
print("Output dimensions:", output_dim)

# 确保 width 参数一致
width = [input_dim, input_dim + 1, output_dim]
print("Width:", width)

# 训练模型
model = KAN(width=width, grid=5, k=3, seed=1, device=device)
model.fit(dataset, opt="LBFGS", steps=20, lamb=0.002, lamb_entropy=2., save_fig=True, img_folder='KAN_picture')
model = model.prune(1e-2)
model.plot('KAN_picture')

# 创建保存模型的文件夹
checkpoint_dir = './KAN_model'
os.makedirs(checkpoint_dir, exist_ok=True)

# 保存模型检查点
checkpoint_path = os.path.join(checkpoint_dir, 'mark')
print(f"Saving model checkpoint to '{checkpoint_path}'")
model.saveckpt(checkpoint_path)
print(f"Model training complete and saved as checkpoint '{checkpoint_path}'")

# 加载模型检查点
loaded_model = KAN.loadckpt(checkpoint_path)
loaded_model = loaded_model.to(device)
loaded_model.eval()
print("Model loaded and set to evaluation mode successfully")