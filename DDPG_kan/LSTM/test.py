import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from new_sumo_env import SumoEnv
import traci
# Step 1: 读取和合并数据
def load_data(data_path):
    all_data = []
    file_list = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
    
    # 遍历文件列表，读取每个文件
    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        data = pd.read_csv(file_path)
        
        # 添加一个新列，标记当前数据属于哪个回合
        data['episode'] = int(file_name.split('-')[0])  # 文件名格式为 "1-lstm_input.csv"，取出"1"作为回合标识
        
        # 将数据加入列表
        all_data.append(data)
    
    # 将所有数据合并成一个大的DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    
    return combined_data

def preprocess_data(data):
    # 去除缺失值
    data = data.dropna()

    # 确保数据集中没有 NaN 或 Inf 值
    if data.isnull().values.any() or np.isinf(data.values).any():
        raise ValueError("Data contains NaN or infinite values.")

    # 提取特征和标签
    features = data[['speed_x', 'speed_y', 'acc', 'front_vehicle_distance', 'front_vehicle_to_speed', 'rear_vehicle_distance', 'rear_vehicle_to_speed', 'Et']]
    labels = data[['acc']].shift(-1)  # 使用当前状态预测下一步的加速度

    # 去掉最后一行，因为它没有对应的下一个加速度值
    features = features[:-1]
    labels = labels[:-1]

    # 对输入特征进行归一化
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = feature_scaler.fit_transform(features)

    # 对输出标签（加速度）进行归一化
    label_scaler = MinMaxScaler(feature_range=(-1, 1))
    labels_scaled = label_scaler.fit_transform(labels)

    # 返回归一化后的输入特征和输出标签，以及两个scaler
    return features_scaled, labels_scaled, feature_scaler, label_scaler

# Step 3: 创建LSTM输入序列
def create_sequences(features, labels, seq_length):
    X = []
    y = []
    
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])  # 提取特征的序列
        y.append(labels[i+seq_length])  # 对应的标签（加速度）
    
    return np.array(X), np.array(y)

# 模型预测并进行反归一化
def evaluate_model(model, test_loader, label_scaler):
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # 对预测的加速度值和实际的加速度值进行反归一化
    predicted_acceleration = label_scaler.inverse_transform(predictions)
    actual_acceleration = label_scaler.inverse_transform(actuals)
    
    return predicted_acceleration, actual_acceleration

# 早停函数
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Step 4: 定义LSTM训练模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if len(lstm_out.shape) == 3:
            out = self.fc(lstm_out[:,-1,:])
        else:
            out = self.fc(lstm_out)
        return out

# LSTM部署加载函数    
class LSTMController:
    def __init__(self, model_path, input_size, hidden_size, output_size, num_layers=2, label_scaler=None):
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 设置为评估模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 用于反归一化的scaler
        self.label_scaler = label_scaler
    
    def predict_acceleration(self, state_sequence):
        # 将状态序列转换为Tensor并转移到正确设备
        state_sequence = torch.tensor(state_sequence).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            acceleration = self.model(state_sequence)  # 预测归一化的加速度
        
        # 将归一化的加速度转换回原始加速度范围（反归一化）
        acceleration = acceleration.cpu().numpy()  # 转换为numpy数组
        if self.label_scaler:
            # 使用label_scaler进行反归一化
            acceleration = self.label_scaler.inverse_transform(acceleration)

        # 将结果限制在动作空间范围内 [-3.0, 4.0]
        acceleration = np.clip(acceleration.item(), -3.0, 4.0).astype(np.float32)   
        return acceleration.reshape((1,)).astype(np.float32)   # 返回标量值


# Step 5: 训练函数
# 在训练代码中加入 early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200):
    early_stopping = EarlyStopping(patience=20, delta=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证集的损失计算
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # 打印损失并检查早停条件
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        if early_stopping(val_loss):
            print("Early stopping")
            break
    
    print("Training completed.")

# 模型训练函数
def train_main():
    # 数据路径
    data_path = "../networks/output_data/output_default/lstm_input"
    # Step 1: 读取和合并数据
    combined_data = load_data(data_path)
    # Step 2: 数据预处理
    features, labels, feature_scaler, label_scaler = preprocess_data(combined_data)
    # Step 3: 创建LSTM输入序列
    seq_length = 20  # 假设使用10个时间步长的序列来进行预测
    X, y = create_sequences(features, labels, seq_length)

    # Step 4: 数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Step 5: 转换为PyTorch的TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    # Step 6: 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Step 7: 初始化模型、损失函数和优化器
    input_size = X_train.shape[2]  # 输入特征的维度
    hidden_size = 256  # LSTM的隐藏层大小
    output_size = 1  # 输出是一个标量（加速度）
    num_layers = 4  # LSTM的层数

    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    model.to(device)

    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3)

    # Step 8: 训练模型
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=800)
    # 保存模型
    model_path = 'model/lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # Step 9: 测试模型
    predictions, actuals = evaluate_model(model, test_loader, label_scaler)


    # Step 10: 对模型输出反归一化
    predicted_acceleration = label_scaler.inverse_transform(predictions)
    actual_acceleration = label_scaler.inverse_transform(actuals)
    print(f"Predicted Acceleration: {predicted_acceleration}")
    print(f"Actual Acceleration: {actual_acceleration}")

# 模型部署sumo函数
def test_main(steps=200):
    model_path = 'model/lstm_model.pth'
    input_size = 8  # 输入特征的维度
    hidden_size = 256  # LSTM的隐藏层大小
    output_size = 1  # 输出是一个标量（加速度）
    num_layers = 4  # LSTM的层数
    controller = LSTMController(model_path, input_size, hidden_size, output_size, num_layers)
    env = SumoEnv(render_mode="rgb_array",max_episodes=steps,flash_episode=True)
    for episode in range (steps):
        state, _ = env.reset()
        done = False
        while not done:
            action = controller.predict_acceleration(state)
            next_state,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            state = next_state
        print("回合==============>>>>>>:",episode)   
    env.close()

# 主函数
if __name__ == "__main__":
    
    # train_main()
    test_main(steps=200)
